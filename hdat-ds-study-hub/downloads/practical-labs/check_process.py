"""공개 자가검사. 공식 hidden test나 완전한 정답 증명은 아닙니다."""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from PIL import Image
import torch
from torch import nn


def equal(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-7, equal_nan=True)


def raises(kind, fn):
    try:
        fn()
    except kind:
        return
    raise AssertionError(f"{kind.__name__} 예외가 필요합니다")


def check(case, m):
    if case in {"P01", "P02"}:
        f = m.scale_sensor_columns if case == "P01" else m.robust_sensor_columns
        df = pd.DataFrame({"a": [4., 10., 7., np.nan], "b": [8, 8, 8, 8], "label": list("ABCD")}, index=[9, 9, 2, 4])
        before = df.copy(deep=True)
        actual = f(df, ["a", "b"])
        expected = [0, 1, .5, np.nan] if case == "P01" else [-1, 1, 0, np.nan]
        equal(actual.a, expected); equal(actual.b, [0, 0, 0, 0])
        assert_frame_equal(df, before)
        assert_frame_equal(actual[["label"]], df[["label"]])
        assert str(actual.a.dtype) == "float64" and str(actual.b.dtype) == "float64"
        assert_frame_equal(f(df, []), df)
        assert f(df.iloc[:0], ["a"]).shape == (0, 3)
        assert f(pd.DataFrame({"a": [np.nan, np.nan]}), ["a"]).a.isna().all()
        raises(KeyError, lambda: f(df, ["missing"]))
        raises(ValueError, lambda: f(df, ["a", "a"]))
        if case == "P01":
            equal(f(pd.DataFrame({"a": [1e9, 1e9 + .25]}), ["a"]).a, [0, 1])
        else:
            equal(f(pd.DataFrame({"a": [0, 0, 0, 0, 100]}), ["a"]).a, [0] * 5)
            equal(f(pd.DataFrame({"a": [0, 2, 4, 6]}), ["a"]).a, [-1, -1/3, 1/3, 1])
    elif case == "P03":
        f = m.add_prior_average
        df = pd.DataFrame({"g": ["A", "B", "A", "A", "B"], "t": [3, 2, 1, 2, 1], "v": [30., 100., 10., 20., 80.], "_row": [9]*5}, index=[7, 7, 2, 2, 1])
        before = df.copy(deep=True)
        out = f(df, "g", "t", "v")
        equal(out.v_prior_mean, [15, 80, np.nan, 10, np.nan])
        assert_frame_equal(df, before); assert_frame_equal(out[df.columns], df)
        tie = pd.DataFrame({"g": ["A"]*4, "t": [1, 1, 2, 3], "v": [10., np.nan, 30., 40.]})
        equal(f(tie, "g", "t", "v").v_prior_mean, [np.nan, 10, 10, 30])
        assert len(f(tie.iloc[:0], "g", "t", "v")) == 0
        for bad in [0, -1, True, 1.2]:
            raises(ValueError, lambda: f(df, "g", "t", "v", bad))
        raises(ValueError, lambda: f(out, "g", "t", "v"))
    elif case == "P04":
        f = m.mean_logit_loss
        x, y = np.array([[0., 0.], [np.log(3), 0.]]), np.array([0, 1])
        before = x.copy()
        assert isinstance(f(x, y), float)
        equal(f(x, y), (np.log(2) + np.log(4)) / 2)
        equal(f(x + 10000, y), f(x, y)); equal(x, before)
        equal(f(np.full((2, 3), 1000.), np.array([0, 2])), np.log(3))
        equal(f(np.array([[9.]]), np.array([0])), 0.)
        for bad in [np.array([.0, 1.]), np.array([[0], [1]]), np.array([0, 2]), np.array([True, False])]:
            raises(ValueError, lambda: f(x, bad))
        raises(ValueError, lambda: f(np.empty((0, 3)), np.array([], dtype=int)))
        raises(ValueError, lambda: f(np.array([[np.inf, 0.]]), np.array([0])))
    elif case == "P05":
        f = m.center_sensor_crop
        for channels in [None, 3, 4]:
            shape = (6, 9) if channels is None else (6, 9, channels)
            raw = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
            img = Image.fromarray(raw)
            out = f(img, 4, 3)
            equal(out, raw[1:4, 2:6]); assert out.flags.writeable and out.dtype == raw.dtype
            out.flat[0] = 0
            equal(np.asarray(img), raw)
        raises(ValueError, lambda: f(img, 10, 3))
        raises(ValueError, lambda: f(img, True, 3))
        raises(TypeError, lambda: f(raw, 2, 2))
        raises(ValueError, lambda: f(Image.new("P", (2, 2)), 1, 1))
    elif case in {"P06", "P07", "P08"}:
        torch.manual_seed(31)
        if case == "P06":
            net, x, single, count = m.SensorMLP(), torch.randn(3, 12), torch.randn(1, 12), 384
            assert [type(v) for v in net.network] == [nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Dropout, nn.Linear]
            assert net.network[3].p == .2
            output = 4
        elif case == "P07":
            net, x, single, count = m.SurfaceCNN(), torch.randn(2, 1, 13, 17), torch.randn(1, 1, 4, 7), 638
            assert [type(v) for v in net.features] == [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.Conv2d, nn.ReLU, nn.AdaptiveAvgPool2d]
            c1, c2 = net.features[0], net.features[4]
            assert c1.bias is None and c1.kernel_size == (3, 3) and c1.padding == (1, 1)
            assert c2.stride == (2, 2) and c2.padding == (1, 1) and c2.bias is not None
            assert net.features[3].kernel_size == 2 and net.features[3].stride == 2
            output = 2
        else:
            net, x, single, count = m.TripGRU(4, 5, 3), torch.randn(2, 7, 4), torch.randn(1, 1, 4), 873
            assert net.gru.batch_first and net.gru.bidirectional and net.gru.num_layers == 2 and net.gru.dropout == .1
            net.eval()
            _, hidden = net.gru(x)
            equal(net(x).detach(), net.head(torch.cat([hidden[-2], hidden[-1]], 1)).detach())
            output = 3
            raises(ValueError, lambda: m.TripGRU(0, 5, 3))
        assert sum(p.numel() for p in net.parameters()) == count
        net.train(); pred = net(x)
        assert pred.shape == (len(x), output) and torch.isfinite(pred).all()
        pred.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        net.eval()
        with torch.no_grad():
            assert net(single).shape == (1, output)
            equal(net(single), net(single))
    else:
        raise ValueError(case)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default="process_starter.py")
    parser.add_argument("--case", default="all", choices=["all"] + [f"P{i:02}" for i in range(1, 9)])
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("student", Path(args.file).resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for case in ([f"P{i:02}" for i in range(1, 9)] if args.case == "all" else [args.case]):
        check(case, module)
        print(f"{case}: 공개 자가검사 PASS (공식 채점과 무관)")
