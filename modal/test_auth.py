"""鉴权逻辑离线单测：不触发 GPU、不产生费用。

    python modal/test_auth.py
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("H3_VOLUME_NAME", "dummy-volume")

import modal  # noqa: E402

# modal.Secret.from_name 在未开启鉴权时不会被调用，这里给个兜底以便离线 import
modal.Secret = types.SimpleNamespace(
    from_name=lambda name: object(),
    __getattr__=lambda name: None,
)

import comfy_endpoint as ce  # noqa: E402


class FakeRequest:
    def __init__(self, token=None):
        self.headers = {} if token is None else {"x-auth-token": token}


def run():
    results = []

    # 1) 未开启鉴权 -> 放行（即使没有 token）
    ok1 = ce.check_auth(FakeRequest(None), require_auth=False) is True
    results.append(("未开启鉴权 + 无 token -> 应放行", ok1))

    # 2) 开启鉴权 + 无 token -> 拒绝
    ok2 = ce.check_auth(FakeRequest(None), require_auth=True, expected_token="SECRET-ABC") is False
    results.append(("开启鉴权 + 无 token -> 应拒绝", ok2))

    # 3) 开启鉴权 + 错误 token -> 拒绝
    ok3 = ce.check_auth(FakeRequest("WRONG"), require_auth=True, expected_token="SECRET-ABC") is False
    results.append(("开启鉴权 + 错误 token -> 应拒绝", ok3))

    # 4) 开启鉴权 + 正确 token -> 放行
    ok4 = ce.check_auth(FakeRequest("SECRET-ABC"), require_auth=True, expected_token="SECRET-ABC") is True
    results.append(("开启鉴权 + 正确 token -> 应放行", ok4))

    # 5) 服务端未配置 token（空）+ 空 token -> 拒绝（防空值绕过）
    ok5 = ce.check_auth(FakeRequest(""), require_auth=True, expected_token="") is False
    results.append(("服务端未配 token + 空 token -> 应拒绝", ok5))

    # 6) headers 缺失也不能崩
    class NoHeaders:
        pass
    ok6 = ce.check_auth(NoHeaders(), require_auth=True, expected_token="SECRET-ABC") is False
    results.append(("请求无 headers -> 应拒绝且不抛异常", ok6))

    # 7) 环境变量路径（模拟容器内从 Secret 注入）
    os.environ["H3_ENDPOINT_TOKEN"] = "ENV-TOKEN"
    ok7 = ce.check_auth(FakeRequest("ENV-TOKEN"), require_auth=True) is True
    results.append(("从环境变量读取 token -> 应放行", ok7))

    width = max(len(name) for name, _ in results)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name.ljust(width)}")

    allok = all(ok for _, ok in results)
    print("\n结果:", f"{sum(1 for _, ok in results if ok)}/{len(results)} 通过",
          "=> ALL PASS" if allok else "=> 有失败用例")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(run())
