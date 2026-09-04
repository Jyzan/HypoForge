from pathlib import Path


WEB_DIR = Path(__file__).resolve().parents[1] / "hypoforge" / "web"


def test_new_question_hero_exposes_pointer_driven_3d_lighting():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    css = (WEB_DIR / "ui-polish.css").read_text(encoding="utf-8")

    assert 'id="heroCore"' in html
    assert 'class="hero-core__depth"' in html
    assert 'class="hero-core__specular"' in html
    assert "function computeHeroLighting" in html
    assert "function setupHeroLighting" in html
    assert "--hero-shadow-x" in html
    assert "--hero-shadow-y" in html
    assert "--hero-tilt-x" in html
    assert "--hero-tilt-y" in html
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert ".hero-core__specular" in css


def test_new_question_background_uses_ping_pong_webgl_fluid_state():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    fluid_js = (WEB_DIR / "hero-fluid.js").read_text(encoding="utf-8")
    css = (WEB_DIR / "ui-polish.css").read_text(encoding="utf-8")
    source = html + fluid_js

    assert 'id="heroFlowCanvas"' in html
    assert "three@0.149.0/build/three.min.js" in html
    assert 'src="/assets/hero-fluid.js"' in html
    assert "function createHeroFluidRenderer" in source
    assert "function createHeroFluidTarget" in source
    assert "function resizeHeroFluidRenderer" in source
    assert "function injectHeroFluid" in source
    assert "function stepHeroFluid" in source
    assert "new THREE.WebGLRenderTarget" in source
    assert "uPreviousState" in source
    assert "fluid.readTarget = fluid.writeTarget" in source
    assert 'main.dataset.heroActive = "true"' in html
    assert "new MutationObserver" in html
    assert ".hero-flow-canvas" in css
    assert ".hero-flow-canvas.is-fallback" in css
    assert '#main[data-hero-active="true"] .hero-flow-canvas' in css
    assert '#main:not([data-hero-active="true"]) .hero-flow-canvas' in css
    assert "opacity: .7" in css
    assert "--hero-bg-x" not in html
    assert "--hero-parallax-x" not in html


def test_hero_fluid_uses_pressure_wave_refraction_instead_of_paint_dye():
    fluid_js = (WEB_DIR / "hero-fluid.js").read_text(encoding="utf-8")

    assert "THREE.HalfFloatType" in fluid_js
    assert "THREE.NearestFilter" in fluid_js
    assert "float pressure = data.x" in fluid_js
    assert "float velocity = data.y" in fluid_js
    assert "float gradientX" in fluid_js
    assert "float gradientY" in fluid_js
    assert "uContentTexture" in fluid_js
    assert "vec2 distortion = state.ba" in fluid_js
    assert "function updateHeroFluidContent" in fluid_js
    assert "dye" not in fluid_js.lower()


def test_pressure_wave_has_a_subtle_cyan_violet_tint_only_when_active():
    fluid_js = (WEB_DIR / "hero-fluid.js").read_text(encoding="utf-8")

    assert "vec3 cyanTint" in fluid_js
    assert "vec3 violetTint" in fluid_js
    assert "float tintStrength = waveActivity" in fluid_js
    assert "mix(cyanTint, violetTint" in fluid_js
