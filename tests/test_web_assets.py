from bashgym import web_assets


def test_installed_assets_preferred_over_checkout(tmp_path, monkeypatch):
    package = tmp_path / "site-packages" / "bashgym"
    assets = package / "web_assets"
    assets.mkdir(parents=True)
    (assets / "index.html").write_text("<main>Studio</main>")
    monkeypatch.setattr(web_assets, "__file__", str(package / "web_assets.py"))
    assert web_assets.frontend_directory() == assets
