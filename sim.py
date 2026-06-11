import argparse
import importlib
import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image

import odl

try:
    from odl import tomo
except ImportError:
    tomo = importlib.import_module("odl.applications.tomo")


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
CSV_PATH = SRC_DIR / "xray_characteristic_data.csv"


@dataclass
class SimulationConfig:
    pixel_size: float
    output_size: int = 512
    E0: int = 40
    metal_name: str = "Titanium"
    metal_density: float = 6.0
    noise_scale: float = 12.0
    filter_name: str = "Hamming"
    freqscale: float = 1.0
    mu_air: float = 0.0
    T1: float = 100.0
    T2: float = 1500.0
    angle_size: float = 0.1
    angle_num: int = 1000
    SOD: float = 50.0
    polynomial_order_for_correction: int = 3
    energy_composition: np.ndarray = field(
        default_factory=lambda: np.arange(1, 121, dtype=np.int32)
    )

    def to_json_dict(self):
        data = asdict(self)
        data["energy_composition"] = self.energy_composition.tolist()
        data["mu_water"] = float(self.mu_water)
        return data


def load_xray_characteristic_data(csv_path=CSV_PATH):
    table = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float32)
    data = {name: np.asarray(table[name], dtype=np.float32) for name in table.dtype.names}
    data["Energy"] = np.asarray(table["Energy"], dtype=np.int32)
    return data


def set_config_for_artifact_simulation(pixel_size, csv_path=CSV_PATH):
    config = SimulationConfig(pixel_size=float(pixel_size))
    config.data = load_xray_characteristic_data(csv_path)
    config.mu_water = float(config.data["Water"][config.E0 - 1])
    return config


def hu2mu(hu, mu_water, mu_air):
    hu = np.asarray(hu, dtype=np.float32)
    return hu / 1000.0 * (mu_water - mu_air) + mu_water


def mu2hu(mu, mu_water, mu_air):
    mu = np.asarray(mu, dtype=np.float32)
    return (mu - mu_water) / (mu_water - mu_air) * 1000.0


def threshold_based_weighting(image, t1, t2):
    img = np.asarray(image, dtype=np.float32)
    w_bone = np.clip((img - t1) / (t2 - t1), 0.0, 1.0)
    bone = w_bone * img
    w_water = np.clip((t2 - img) / (t2 - t1), 0.0, 1.0)
    water = w_water * img
    return water, bone


def create_phantom(xsize, ysize, radius, mu_water):
    yy, xx = np.ogrid[:ysize, :xsize]
    cx = (xsize - 1) / 2.0
    cy = (ysize - 1) / 2.0
    phantom = ((xx - cx) ** 2 + (yy - cy) ** 2) < radius ** 2
    return phantom.astype(np.float32) * np.float32(mu_water)


def resize_to_shape(array, shape, is_mask=False):
    if tuple(array.shape[:2]) == tuple(shape):
        return np.asarray(array)
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    pil = Image.fromarray(np.asarray(array))
    resized = pil.resize((shape[1], shape[0]), resample=resample)
    return np.asarray(resized)


def ensure_matching_shape(image, metal_mask):
    image = np.asarray(image)
    metal_mask = np.asarray(metal_mask)
    if image.shape != metal_mask.shape:
        # MATLAB 逻辑要求掩膜与图像同尺寸，否则金属区域置零会直接报错。
        metal_mask = resize_to_shape(metal_mask, image.shape, is_mask=True)
    return image, metal_mask


def _get_filter_name(filter_name):
    mapping = {
        "Ram-Lak": "Ram-Lak",
        "Shepp-Logan": "Shepp-Logan",
        "Cosine": "Cosine",
        "Hann": "Hann",
        "Hamming": "Hamming",
        "None": None,
    }
    return mapping.get(filter_name, filter_name)


def build_geometry(config):
    size_cm = config.output_size * config.pixel_size
    reco_space = odl.uniform_discr(
        min_pt=[-size_cm / 2.0, -size_cm / 2.0],
        max_pt=[size_cm / 2.0, size_cm / 2.0],
        shape=[config.output_size, config.output_size],
        dtype="float32",
    )

    angle_partition = odl.uniform_partition(0.0, 2.0 * np.pi, config.angle_num)

    detector_spacing = config.SOD * math.radians(config.angle_size)
    detector_span = 2.0 * math.sqrt(2.0) * size_cm
    detector_count = int(math.ceil(detector_span / detector_spacing)) + 1
    detector_partition = odl.uniform_partition(
        min_pt=-(detector_span / 2.0),
        max_pt=(detector_span / 2.0),
        shape=detector_count,
    )

    geometry = tomo.FanBeamGeometry(
        apart=angle_partition,
        dpart=detector_partition,
        src_radius=config.SOD,
        det_radius=config.SOD,
    )

    last_error = None
    for impl in ("astra_cuda", "astra_cpu"):
        try:
            fp_op = tomo.RayTransform(reco_space, geometry, impl=impl)
            break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError("Unable to build ODL ray transform") from last_error

    fbp_op = tomo.fbp_op(
        fp_op,
        filter_type=_get_filter_name(config.filter_name),
        frequency_scaling=config.freqscale,
    )
    return fp_op, fbp_op


def _poisson_noisy_intensity(intensity, noise_scale):
    del noise_scale
    noisy = np.random.poisson(np.clip(intensity, 0.0, None)).astype(np.float32)
    noisy[noisy <= 0] = 1.0
    return noisy


def _poly_projection(d_water, d_bone, d_metal, config):
    total_intensity = 0.0
    poly_y = np.zeros_like(d_water, dtype=np.float32)

    m0_water = float(config.data["Water"][config.E0 - 1])
    m0_bone = float(config.data["Bone"][config.E0 - 1])
    m0_metal = float(config.data[config.metal_name][config.E0 - 1])

    for energy in np.asarray(config.energy_composition).ravel():
        energy_idx = int(energy) - 1
        intensity = float(config.data["Intensity"][energy_idx])
        if intensity <= 0:
            continue

        m_water = float(config.data["Water"][energy_idx])
        m_bone = float(config.data["Bone"][energy_idx])
        m_metal = float(config.data[config.metal_name][energy_idx])

        drr = (
            d_water * (m_water / m0_water)
            + d_bone * (m_bone / m0_bone)
            + d_metal * (m_metal / m0_metal)
        )
        poly_y += intensity * np.exp(-drr)
        total_intensity += intensity

    poly_y = np.clip(poly_y, np.finfo(np.float32).tiny, None)
    return poly_y, total_intensity


def phantom_proj_mono(phantom, config, fp_op=None):
    if fp_op is None:
        fp_op, _ = build_geometry(config)
    return np.asarray(fp_op(np.asarray(phantom, dtype=np.float32)), dtype=np.float32)


def phantom_proj_poly(phantom, config, fp_op=None):
    if fp_op is None:
        fp_op, _ = build_geometry(config)
    d_water = np.asarray(fp_op(np.asarray(phantom, dtype=np.float32)), dtype=np.float32)
    poly_y, total_intensity = _poly_projection(
        d_water,
        np.zeros_like(d_water, dtype=np.float32),
        np.zeros_like(d_water, dtype=np.float32),
        config,
    )
    return -np.log(poly_y / total_intensity)


def _fit_correction_polynomial(p_poly, p_mono, degree):
    x = np.asarray(p_poly, dtype=np.float64).ravel()
    y = np.asarray(p_mono, dtype=np.float64).ravel()

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        raise ValueError("No valid samples available for correction fit.")

    unique_x = np.unique(x)
    fit_degree = max(0, min(int(degree), unique_x.size - 1))

    # Fit in a normalized domain and convert back to the power basis expected by np.polyval.
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.RankWarning)
        while fit_degree > 0:
            try:
                poly = np.polynomial.Polynomial.fit(x, y, deg=fit_degree)
                return poly.convert().coef[::-1].astype(np.float32)
            except np.exceptions.RankWarning:
                fit_degree -= 1

    poly = np.polynomial.Polynomial.fit(x, y, deg=fit_degree)
    return poly.convert().coef[::-1].astype(np.float32)


def water_correction(phantom, config, fp_op=None):
    p_mono = phantom_proj_mono(phantom, config, fp_op=fp_op)
    p_poly = phantom_proj_poly(phantom, config, fp_op=fp_op)
    return _fit_correction_polynomial(
        p_poly,
        p_mono,
        config.polynomial_order_for_correction,
    )


def metal_artifact_simulation(image, x_metal, config, fp_op=None, fbp_op=None):
    image, x_metal = ensure_matching_shape(image, x_metal)

    if fp_op is None or fbp_op is None:
        fp_op, fbp_op = build_geometry(config)

    t1 = hu2mu(config.T1, config.mu_water, config.mu_air)
    t2 = hu2mu(config.T2, config.mu_water, config.mu_air)
    x_water, x_bone = threshold_based_weighting(image, t1, t2)

    metal_mask = (np.asarray(x_metal) > 0).astype(np.float32)
    x_water[metal_mask > 0] = 0.0
    x_bone[metal_mask > 0] = 0.0

    mu_metal0 = float(config.data[config.metal_name][config.E0 - 1]) * float(config.metal_density)
    x_metal_mu = metal_mask * mu_metal0

    d_water = np.asarray(fp_op(x_water.astype(np.float32)), dtype=np.float32)
    d_bone = np.asarray(fp_op(x_bone.astype(np.float32)), dtype=np.float32)
    d_metal = np.asarray(fp_op(x_metal_mu.astype(np.float32)), dtype=np.float32)

    poly_y, total_intensity = _poly_projection(d_water, d_bone, d_metal, config)
    noisy_y = _poisson_noisy_intensity(poly_y, config.noise_scale)
    noisy_y = np.clip(noisy_y, np.finfo(np.float32).tiny, None)

    p = -np.log(noisy_y / total_intensity)
    p = np.polyval(np.asarray(config.correction_coeff, dtype=np.float32), p)

    sim = np.asarray(fbp_op(p), dtype=np.float32)
    sim[sim < 0] = 0.0
    return sim


def set_window(image, vmin, vmax):
    image = np.clip(image, vmin, vmax)
    image = (image - vmin) / max(vmax - vmin, 1e-6)
    return (image * 255.0).astype(np.uint8)


def _extract_sample_field(sample, name):
    value = sample[name]
    while isinstance(value, np.ndarray) and value.dtype == object:
        value = value.flat[0]
    return np.asarray(value)


def load_sample_mat(mat_path):
    data = sio.loadmat(mat_path)
    if "sample" in data:
        sample = data["sample"]
        image = _extract_sample_field(sample, "image")
        metal = _extract_sample_field(sample, "metal")
        pixel_size = float(_extract_sample_field(sample, "pixel_size").squeeze())
    else:
        image = np.asarray(data["image"])
        metal = np.asarray(data["metal"])
        pixel_size = float(np.asarray(data["pixel_size"]).squeeze())
    return image.astype(np.float32), metal.astype(np.float32), pixel_size


def run_demo(sample_mat=None, output_dir=None):
    sample_mat = Path(sample_mat or ROOT / "sample_data" / "sample_2.mat")
    output_dir = Path(output_dir or ROOT / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_hu, metal, pixel_size = load_sample_mat(sample_mat)
    config = set_config_for_artifact_simulation(pixel_size)
    fp_op, fbp_op = build_geometry(config)

    image_hu = np.asarray(image_hu, dtype=np.float32)
    image_hu[image_hu < -500] = -1000.0
    image_mu = hu2mu(image_hu, config.mu_water, config.mu_air)

    phantom = create_phantom(
        config.output_size,
        config.output_size,
        200,
        config.mu_water,
    )
    config.correction_coeff = water_correction(phantom, config, fp_op=fp_op)

    sim_mu = metal_artifact_simulation(image_mu, metal, config, fp_op=fp_op, fbp_op=fbp_op)
    sim_hu = mu2hu(sim_mu, config.mu_water, config.mu_air)

    Image.fromarray(set_window(image_hu, -150, 350)).save(output_dir / "input.png")
    Image.fromarray(set_window(sim_hu, -150, 350)).save(output_dir / "output.png")
    np.save(output_dir / "simulation_output.npy", sim_hu.astype(np.float32))

    with open(output_dir / "simulation_config.json", "w", encoding="utf-8") as f:
        json.dump(config.to_json_dict(), f, indent=2)

    return sim_hu


def main():
    parser = argparse.ArgumentParser(description="Metal artifact simulation in Python")
    parser.add_argument("--sample", type=str, default=None, help="Path to MATLAB sample .mat")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    run_demo(args.sample, args.output_dir)


if __name__ == "__main__":
    main()
