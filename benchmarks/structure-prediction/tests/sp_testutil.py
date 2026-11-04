# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Test helpers: synthetic structures and fake host/AWS clients."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS",
    "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN",
    "R": "ARG", "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}  # fmt: skip


def helix(n: int, seed: int = 0) -> np.ndarray:
    """CA trace of an ideal-ish helix (3.8 Å steps), slightly perturbed."""
    t = np.arange(n)
    xyz = np.stack([2.3 * np.cos(t * 1.745), 2.3 * np.sin(t * 1.745), 1.5 * t], axis=1)
    return xyz + np.random.default_rng(seed).normal(0, 0.05, xyz.shape)


def write_ca_cif(
    path: Path,
    seq: str,
    ca: np.ndarray,
    chain: str = "A",
    start: int = 1,
    ligand: dict[str, np.ndarray] | None = None,
) -> Path:
    """Minimal mmCIF with CA atoms (and an optional ligand) that gemmi reads as a polymer."""
    import gemmi

    st = gemmi.Structure()
    model = gemmi.Model("1")
    ch = gemmi.Chain(chain)
    for i, (aa, xyz) in enumerate(zip(seq, ca, strict=True)):
        res = gemmi.Residue()
        res.name = THREE[aa]
        res.seqid = gemmi.SeqId(start + i, " ")
        res.entity_type = gemmi.EntityType.Polymer
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(*map(float, xyz))
        res.add_atom(atom)
        ch.add_residue(res)
    model.add_chain(ch)
    if ligand:
        lch = gemmi.Chain("L")
        res = gemmi.Residue()
        res.name = "LIG"
        res.seqid = gemmi.SeqId(1, " ")
        res.het_flag = "H"
        res.entity_type = gemmi.EntityType.NonPolymer
        for name, xyz in ligand.items():
            atom = gemmi.Atom()
            atom.name = name
            atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(*map(float, xyz))
            res.add_atom(atom)
        lch.add_residue(res)
        model.add_chain(lch)
    st.add_model(model)
    st.setup_entities()
    st.make_mmcif_document().write_file(str(path))
    return path


class FakeHost:
    """Canned command outputs for manifest tests."""

    def __init__(self, kind: str = "neuron") -> None:
        """Pick canned outputs for a neuron or gpu host."""
        self.kind = kind
        self.calls: list[list[str]] = []

    def run(self, argv: list[str]) -> str:
        """Pretend to run argv."""
        self.calls.append(argv)
        cmd = argv[0]
        if cmd == "neuron-ls":
            return json.dumps(
                [{"neuron_device": 0, "nc_count": 8, "memory_size": 103079215104}]
            )
        if cmd == "nvidia-smi" and len(argv) > 1:
            return "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 580.82.07, 00000000:00:1E.0\n"
        if cmd == "nvidia-smi":
            return "| NVIDIA-SMI 580.82.07  Driver Version: 580.82.07  CUDA Version: 13.0 |"
        if cmd == "dpkg-query":
            return f"{argv[-1]}\t2.28.10.0\n"
        if cmd == "git":
            return "0123456789abcdef0123456789abcdef01234567\n"
        if argv[1] == "-c" and "importlib.metadata" in argv[2]:
            name = argv[2].split("m.version(")[1].strip("')\"")
            versions = {
                "torch-neuronx": "2.14.3",
                "torch": "2.14.0",
                "neuronx-cc": "2.28.7631",
            }
            if name in versions:
                return versions[name] + "\n"
            raise RuntimeError("PackageNotFoundError")
        raise FileNotFoundError(cmd)

    def http(self, url: str, headers: dict[str, str]) -> str:
        """Pretend IMDSv2 identity document."""
        assert headers.get("X-aws-ec2-metadata-token") == "tok"
        return json.dumps(
            {
                "instanceType": "trn2.3xlarge"
                if self.kind == "neuron"
                else "g7e.2xlarge",
                "availabilityZone": "us-east-1a",
                "region": "us-east-1",
                "imageId": "ami-0123456789abcdef0",
                "instanceId": "i-0123456789abcdef0",
            }
        )


class FakePricing:
    """boto3 pricing client stand-in."""

    def __init__(self, usd: str = "2.2350000000") -> None:
        """Return one on-demand price."""
        self.usd = usd
        self.kwargs: dict[str, Any] = {}

    def get_products(self, **kwargs: Any) -> dict[str, Any]:
        """Canned GetProducts response."""
        self.kwargs = kwargs
        item = {
            "terms": {
                "OnDemand": {
                    "x": {"priceDimensions": {"y": {"pricePerUnit": {"USD": self.usd}}}}
                }
            }
        }
        return {"PriceList": [json.dumps(item)]}


class FakeEC2:
    """boto3 ec2 client stand-in."""

    def describe_spot_price_history(self, **kwargs: Any) -> dict[str, Any]:
        """Canned spot price."""
        return {
            "SpotPriceHistory": [
                {"SpotPrice": "0.9000", "Timestamp": "2026-11-04T00:00:00Z"}
            ]
        }
