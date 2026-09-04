(function(){
  "use strict";

  const SIMULATION_SCALE = 0.5;
  const WAVE_SPEED = 1.08;
  const WAVE_DAMPING = 0.985;
  const RIPPLE_RADIUS = 19;

  const VERTEX_SHADER = `
    varying vec2 vUv;
    void main(){
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `;

  const SIMULATION_SHADER = `
    precision highp float;
    varying vec2 vUv;
    uniform sampler2D uPreviousState;
    uniform vec2 uResolution;
    uniform vec3 uMouse;
    uniform float uWaveSpeed;
    uniform float uDamping;
    uniform float uRippleRadius;
    uniform float uPressureStrength;

    void main(){
      vec2 texel = 1.0 / uResolution;
      vec2 coordinate = vUv;
      vec4 data = texture2D(uPreviousState, coordinate);
      float pressure = data.x;
      float velocity = data.y;

      float pressureRight = texture2D(uPreviousState, coordinate + vec2(texel.x, 0.0)).x;
      float pressureLeft = texture2D(uPreviousState, coordinate - vec2(texel.x, 0.0)).x;
      float pressureUp = texture2D(uPreviousState, coordinate + vec2(0.0, texel.y)).x;
      float pressureDown = texture2D(uPreviousState, coordinate - vec2(0.0, texel.y)).x;

      if(coordinate.x < texel.x) pressureLeft = pressureRight;
      if(coordinate.x > 1.0 - texel.x) pressureRight = pressureLeft;
      if(coordinate.y < texel.y) pressureDown = pressureUp;
      if(coordinate.y > 1.0 - texel.y) pressureUp = pressureDown;

      velocity += uWaveSpeed * (-2.0 * pressure + pressureRight + pressureLeft) / 3.0;
      velocity += uWaveSpeed * (-2.0 * pressure + pressureUp + pressureDown) / 3.0;
      pressure += uWaveSpeed * velocity * 1.2;
      velocity -= 0.001 * uWaveSpeed * pressure;
      velocity *= 1.0 - 0.005 * uWaveSpeed;
      pressure *= uDamping;

      float gradientX = (pressureRight - pressureLeft) * 0.5;
      float gradientY = (pressureUp - pressureDown) * 0.5;

      if(uMouse.z > 0.5){
        float distanceFromMouse = distance(coordinate * uResolution, uMouse.xy);
        if(distanceFromMouse <= uRippleRadius){
          float falloff = 1.0 - distanceFromMouse / uRippleRadius;
          pressure += falloff * falloff * uPressureStrength;
        }
      }

      gl_FragColor = vec4(pressure, velocity, gradientX, gradientY);
    }
  `;

  const DISPLAY_SHADER = `
    precision highp float;
    varying vec2 vUv;
    uniform sampler2D uState;
    uniform sampler2D uContentTexture;
    uniform float uTime;

    void main(){
      vec4 state = texture2D(uState, vUv);
      vec2 distortion = state.ba * 0.29;
      vec2 distortedUv = clamp(vUv + distortion, vec2(0.002), vec2(0.998));
      vec3 content = texture2D(uContentTexture, distortedUv).rgb;

      float waveHeight = abs(state.x);
      float gradientX = state.b;
      float gradientY = state.a;
      vec3 normal = normalize(vec3(-gradientX * 4.8, 0.34, -gradientY * 4.8));
      vec3 lightDirection = normalize(vec3(-2.5, 6.0, 2.5));
      float specular = pow(max(0.0, dot(normal, lightDirection)), 180.0);
      float broadSpecular = pow(max(0.0, dot(normal, lightDirection)), 70.0);
      float caustic = (sin(waveHeight * 23.0 + uTime * 0.16) * 0.5 + 0.5) * waveHeight;

      float slope = length(state.ba);
      float waveActivity = smoothstep(0.0015, 0.07, slope * 2.8 + waveHeight * 0.12);
      vec3 cyanTint = vec3(0.08, 0.67, 0.86);
      vec3 violetTint = vec3(0.43, 0.24, 0.8);
      float tintMix = clamp(0.46 + state.x * 2.6 + (vUv.x - 0.5) * 0.24, 0.0, 1.0);
      vec3 waveTint = mix(cyanTint, violetTint, tintMix);
      float tintStrength = waveActivity * 0.36;
      vec3 color = mix(content, waveTint, tintStrength);
      color += vec3(1.0) * specular * 0.95;
      color += vec3(0.56, 0.82, 1.0) * broadSpecular * 0.26;
      color += vec3(0.68, 0.88, 1.0) * caustic * 0.15;

      float alpha = waveActivity * (0.11 + min(waveHeight * 0.08 + slope * 2.8, 0.29));
      alpha = clamp(alpha + specular * 0.32, 0.0, 0.48);
      gl_FragColor = vec4(color, alpha);
    }
  `;

  function createHeroFluidTarget(width, height, type){
    return new THREE.WebGLRenderTarget(width, height, {
      minFilter:THREE.NearestFilter,
      magFilter:THREE.NearestFilter,
      format:THREE.RGBAFormat,
      type:type || THREE.HalfFloatType,
      depthBuffer:false,
      stencilBuffer:false,
    });
  }

  function updateHeroFluidContent(fluid){
    const canvas = fluid.contentCanvas;
    const context = fluid.contentContext;
    const width = Math.max(2, Math.round(fluid.width));
    const height = Math.max(2, Math.round(fluid.height));
    canvas.width = width;
    canvas.height = height;

    const base = context.createLinearGradient(0, height * 0.15, width, height * 0.82);
    base.addColorStop(0, "#d8f8fb");
    base.addColorStop(0.34, "#b9edf6");
    base.addColorStop(0.68, "#c6cdfc");
    base.addColorStop(1, "#e2d7fb");
    context.fillStyle = base;
    context.fillRect(0, 0, width, height);

    context.save();
    context.globalCompositeOperation = "screen";
    context.filter = "blur(18px)";
    context.lineCap = "round";
    for(let row = -2; row < 10; row += 1){
      const y = height * (row / 8);
      context.beginPath();
      context.moveTo(-width * 0.1, y);
      context.bezierCurveTo(
        width * 0.22, y + height * 0.09,
        width * 0.58, y - height * 0.08,
        width * 1.1, y + height * 0.035
      );
      context.strokeStyle = row % 2
        ? "rgba(255,255,255,0.22)"
        : "rgba(125,211,252,0.14)";
      context.lineWidth = Math.max(18, height * 0.045);
      context.stroke();
    }
    context.restore();
    fluid.contentTexture.needsUpdate = true;
  }

  function clearHeroFluidTargets(fluid){
    const renderer = fluid.renderer;
    const previousTarget = renderer.getRenderTarget();
    const previousAlpha = renderer.getClearAlpha();
    const previousColor = renderer.getClearColor(new THREE.Color()).clone();
    renderer.setClearColor(0x000000, 0);
    for(const target of [fluid.readTarget, fluid.writeTarget]){
      renderer.setRenderTarget(target);
      renderer.clear(true, false, false);
    }
    renderer.setRenderTarget(previousTarget);
    renderer.setClearColor(previousColor, previousAlpha);
  }

  function createHeroFluidRenderer(canvas){
    if(!canvas || !window.THREE){
      canvas?.classList.add("is-fallback");
      return null;
    }
    try{
      const renderer = new THREE.WebGLRenderer({
        canvas,
        alpha:true,
        antialias:false,
        powerPreference:"low-power",
        premultipliedAlpha:false,
      });
      renderer.setClearColor(0x000000, 0);

      const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
      const geometry = new THREE.PlaneGeometry(2, 2);
      const simulationScene = new THREE.Scene();
      const displayScene = new THREE.Scene();
      const contentCanvas = document.createElement("canvas");
      const contentContext = contentCanvas.getContext("2d", {alpha:false});
      const contentTexture = new THREE.CanvasTexture(contentCanvas);
      contentTexture.minFilter = THREE.LinearFilter;
      contentTexture.magFilter = THREE.LinearFilter;

      const simulationMaterial = new THREE.ShaderMaterial({
        uniforms:{
          uPreviousState:{value:null},
          uResolution:{value:new THREE.Vector2(1, 1)},
          uMouse:{value:new THREE.Vector3(-1, -1, 0)},
          uWaveSpeed:{value:WAVE_SPEED},
          uDamping:{value:WAVE_DAMPING},
          uRippleRadius:{value:RIPPLE_RADIUS},
          uPressureStrength:{value:1},
        },
        vertexShader:VERTEX_SHADER,
        fragmentShader:SIMULATION_SHADER,
        depthTest:false,
        depthWrite:false,
      });
      const displayMaterial = new THREE.ShaderMaterial({
        uniforms:{
          uState:{value:null},
          uContentTexture:{value:contentTexture},
          uTime:{value:0},
        },
        vertexShader:VERTEX_SHADER,
        fragmentShader:DISPLAY_SHADER,
        transparent:true,
        blending:THREE.NormalBlending,
        depthTest:false,
        depthWrite:false,
      });
      simulationScene.add(new THREE.Mesh(geometry, simulationMaterial));
      displayScene.add(new THREE.Mesh(geometry, displayMaterial));

      const fluid = {
        canvas,
        renderer,
        camera,
        geometry,
        simulationScene,
        displayScene,
        simulationMaterial,
        displayMaterial,
        contentCanvas,
        contentContext,
        contentTexture,
        readTarget:null,
        writeTarget:null,
        targetType:THREE.HalfFloatType,
        width:0,
        height:0,
        simWidth:0,
        simHeight:0,
        active:false,
        rafId:0,
        lastFrame:0,
        pointerPending:false,
        pointerX:-1,
        pointerY:-1,
        pressureStrength:1,
      };
      resizeHeroFluidRenderer(fluid, true);
      canvas.classList.remove("is-fallback");
      return fluid;
    }catch(error){
      console.warn("Water ripple renderer unavailable; using CSS fallback.", error);
      canvas.classList.add("is-fallback");
      return null;
    }
  }

  function resizeHeroFluidRenderer(fluid, force){
    if(!fluid?.renderer) return;
    const rect = fluid.canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    if(!force && width === fluid.width && height === fluid.height) return;

    fluid.width = width;
    fluid.height = height;
    fluid.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.25));
    fluid.renderer.setSize(width, height, false);
    updateHeroFluidContent(fluid);

    const scale = width < 720 ? 0.42 : SIMULATION_SCALE;
    const simWidth = Math.max(128, Math.floor(width * scale));
    const simHeight = Math.max(96, Math.floor(height * scale));
    if(simWidth === fluid.simWidth && simHeight === fluid.simHeight) return;

    fluid.readTarget?.dispose();
    fluid.writeTarget?.dispose();
    fluid.readTarget = createHeroFluidTarget(simWidth, simHeight, fluid.targetType);
    fluid.writeTarget = createHeroFluidTarget(simWidth, simHeight, fluid.targetType);
    fluid.simWidth = simWidth;
    fluid.simHeight = simHeight;
    fluid.simulationMaterial.uniforms.uResolution.value.set(simWidth, simHeight);
    clearHeroFluidTargets(fluid);
  }

  function injectHeroFluid(fluid, clientX, clientY, movementX, movementY){
    if(!fluid?.active || !fluid.width || !fluid.height) return;
    const rect = fluid.canvas.getBoundingClientRect();
    const localX = clientX - rect.left;
    const localY = clientY - rect.top;
    if(localX < 0 || localX > rect.width || localY < 0 || localY > rect.height) return;
    const speed = Math.hypot(movementX, movementY);
    if(speed < 0.35) return;

    fluid.pointerX = localX / Math.max(rect.width, 1) * fluid.simWidth;
    fluid.pointerY = (1 - localY / Math.max(rect.height, 1)) * fluid.simHeight;
    fluid.pressureStrength = Math.min(1.35, 0.72 + speed / 42);
    fluid.pointerPending = true;
  }

  function stepHeroFluid(fluid, timestamp){
    fluid.rafId = 0;
    if(!fluid.active) return;
    resizeHeroFluidRenderer(fluid, false);

    const uniforms = fluid.simulationMaterial.uniforms;
    uniforms.uPreviousState.value = fluid.readTarget.texture;
    uniforms.uMouse.value.set(
      fluid.pointerX,
      fluid.pointerY,
      fluid.pointerPending ? 1 : 0
    );
    uniforms.uPressureStrength.value = fluid.pressureStrength;

    fluid.renderer.setRenderTarget(fluid.writeTarget);
    fluid.renderer.render(fluid.simulationScene, fluid.camera);

    const previousReadTarget = fluid.readTarget;
    fluid.readTarget = fluid.writeTarget;
    fluid.writeTarget = previousReadTarget;
    fluid.pointerPending = false;
    uniforms.uMouse.value.z = 0;

    fluid.displayMaterial.uniforms.uState.value = fluid.readTarget.texture;
    fluid.displayMaterial.uniforms.uTime.value = timestamp * 0.001;
    fluid.renderer.setRenderTarget(null);
    fluid.renderer.render(fluid.displayScene, fluid.camera);
    fluid.lastFrame = timestamp;
    fluid.rafId = requestAnimationFrame(nextTimestamp => stepHeroFluid(fluid, nextTimestamp));
  }

  function setHeroFluidActive(fluid, active){
    if(!fluid?.renderer) return;
    fluid.active = Boolean(active);
    if(fluid.active){
      resizeHeroFluidRenderer(fluid, false);
      if(!fluid.rafId) fluid.rafId = requestAnimationFrame(timestamp => stepHeroFluid(fluid, timestamp));
    }else{
      if(fluid.rafId) cancelAnimationFrame(fluid.rafId);
      fluid.rafId = 0;
      fluid.pointerPending = false;
      fluid.renderer.setRenderTarget(null);
      fluid.renderer.clear(true, false, false);
    }
  }

  window.createHeroFluidRenderer = createHeroFluidRenderer;
  window.createHeroFluidTarget = createHeroFluidTarget;
  window.resizeHeroFluidRenderer = resizeHeroFluidRenderer;
  window.injectHeroFluid = injectHeroFluid;
  window.stepHeroFluid = stepHeroFluid;
  window.setHeroFluidActive = setHeroFluidActive;
  window.updateHeroFluidContent = updateHeroFluidContent;
})();
