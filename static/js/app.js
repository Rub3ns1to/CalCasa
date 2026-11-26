(() => {
  const HISTORY_STORAGE_KEY = "tasador_madrid_history_v1";
  const MAX_HISTORY_ITEMS = 8;

  const PRESETS = {
    salamanca: {
      barrio: "Salamanca",
      micro_barrio: "Recoletos",
      tipo_vivienda: "Ático",
      metros_cuadrados: 128,
      habitaciones: 3,
      banos: 2,
      planta: 6,
      ascensor: true,
      exterior: true,
      terraza: true,
      garaje: false,
      aire_acondicionado: true,
      trastero: true,
    },
    arganzuela: {
      barrio: "Arganzuela",
      micro_barrio: "Delicias",
      tipo_vivienda: "Piso",
      metros_cuadrados: 92,
      habitaciones: 3,
      banos: 2,
      planta: 3,
      ascensor: true,
      exterior: true,
      terraza: false,
      garaje: true,
      aire_acondicionado: true,
      trastero: false,
    },
    vallecas: {
      barrio: "Puente de Vallecas",
      micro_barrio: "Numancia",
      tipo_vivienda: "Piso",
      metros_cuadrados: 68,
      habitaciones: 2,
      banos: 1,
      planta: 4,
      ascensor: false,
      exterior: true,
      terraza: false,
      garaje: false,
      aire_acondicionado: false,
      trastero: false,
    },
  };

  const state = {
    toggles: {
      ascensor: true,
      exterior: true,
      terraza: false,
      garaje: false,
      aire_acondicionado: false,
      trastero: false,
    },
    photos: [], // [{ file, url }]
    districtsData: new Map(), // nombre -> { mediana_m2, barrios: [...] }
    maxFiles: 10,
    maxSizeMb: 8,
    maxSizeBytes: 8 * 1024 * 1024,
    maxAdjustmentPct: 15.0,
    modelMeta: null,
    lastAppraisal: null,
    history: [],
    serverVisionMode: "demo",
    serverVisionModel: "",
    serverHealthLoaded: false,
  };

  const els = {
    visionBadge: document.getElementById("vision-mode-badge"),
    openModelModalBtn: document.getElementById("open-model-modal-btn"),
    footerModelModalBtn: document.getElementById("footer-model-modal-btn"),
    closeModelModalBtn: document.getElementById("close-model-modal-btn"),
    modelModal: document.getElementById("model-modal"),
    modalDatasetSubtitle: document.getElementById("modal-dataset-subtitle"),
    modalKpis: document.getElementById("modal-kpis"),
    modalBenchmarkBody: document.getElementById("modal-benchmark-body"),
    modalImportancesList: document.getElementById("modal-importances-list"),
    form: document.getElementById("appraisal-form"),
    barrio: document.getElementById("barrio"),
    microBarrio: document.getElementById("micro_barrio"),
    tipoVivienda: document.getElementById("tipo_vivienda"),
    districtHint: document.getElementById("district-median-hint"),
    metros: document.getElementById("metros_cuadrados"),
    planta: document.getElementById("planta"),
    habitaciones: document.getElementById("habitaciones"),
    banos: document.getElementById("banos"),
    toggleBtns: document.querySelectorAll(".toggle-btn"),
    presetBtns: document.querySelectorAll(".preset-btn"),
    dropZone: document.getElementById("drop-zone"),
    photoInput: document.getElementById("photo-input"),
    photoLimitLabel: document.getElementById("photo-limit-label"),
    photoCounter: document.getElementById("photo-counter"),
    previewGrid: document.getElementById("preview-grid"),
    byokPanel: document.getElementById("byok-panel"),
    byokStatusPill: document.getElementById("byok-status-pill"),
    byokHelperText: document.getElementById("byok-helper-text"),
    byokProvider: document.getElementById("byok-provider"),
    byokApiKey: document.getElementById("byok-api-key"),
    byokToggleVis: document.getElementById("byok-toggle-vis"),
    byokClearBtn: document.getElementById("byok-clear-btn"),
    errorBanner: document.getElementById("error-banner"),
    submitBtn: document.getElementById("submit-btn"),
    submitLabel: document.getElementById("submit-label"),
    emptyState: document.getElementById("empty-state"),
    resultsPanel: document.getElementById("results-panel"),
    warningBanner: document.getElementById("vision-warning-banner"),
    propertySummary: document.getElementById("result-property-summary"),
    comparisonPill: document.getElementById("district-comparison-pill"),
    downloadPdfBtn: document.getElementById("download-pdf-btn"),
    downloadPdfLabel: document.getElementById("download-pdf-label"),
    resultRange: document.getElementById("result-range"),
    resultCentral: document.getElementById("result-central"),
    resultM2: document.getElementById("result-m2"),
    resultM2Range: document.getElementById("result-m2-range"),
    stepBasePrice: document.getElementById("step-base-price"),
    stepBaseM2: document.getElementById("step-base-m2"),
    stepVisualPct: document.getElementById("step-visual-pct"),
    stepVisualEur: document.getElementById("step-visual-eur"),
    stepBandPct: document.getElementById("step-band-pct"),
    stepPhotosCount: document.getElementById("step-photos-count"),
    tabularFactorsCard: document.getElementById("tabular-factors-card"),
    tabularFactorsList: document.getElementById("tabular-factors-list"),
    visionModelCaption: document.getElementById("vision-model-caption"),
    visionEstadoPill: document.getElementById("vision-estado-pill"),
    visionWeightedScore: document.getElementById("vision-weighted-score"),
    scoreReformaVal: document.getElementById("score-reforma-val"),
    scoreCalidadesVal: document.getElementById("score-calidades-val"),
    scoreLuminosidadVal: document.getElementById("score-luminosidad-val"),
    barReforma: document.getElementById("bar-reforma"),
    barCalidades: document.getElementById("bar-calidades"),
    barLuminosidad: document.getElementById("bar-luminosidad"),
    visionJustification: document.getElementById("vision-justification"),
    strengthsList: document.getElementById("strengths-list"),
    improvementsList: document.getElementById("improvements-list"),
    simLuminosidadNote: document.getElementById("sim-luminosidad-note"),
    simReforma: document.getElementById("sim-reforma"),
    simCalidades: document.getElementById("sim-calidades"),
    simReformaLabel: document.getElementById("sim-reforma-label"),
    simCalidadesLabel: document.getElementById("sim-calidades-label"),
    simBtnCurrent: document.getElementById("sim-btn-current"),
    simBtnMid: document.getElementById("sim-btn-mid"),
    simBtnFull: document.getElementById("sim-btn-full"),
    simPrice: document.getElementById("sim-price"),
    simM2: document.getElementById("sim-m2"),
    simDelta: document.getElementById("sim-delta"),
    comparablesCard: document.getElementById("comparables-card"),
    comparablesBody: document.getElementById("comparables-body"),
    historyCard: document.getElementById("history-card"),
    historyList: document.getElementById("history-list"),
    clearHistoryBtn: document.getElementById("clear-history-btn"),
    metricsFooter: document.getElementById("model-metrics-footer"),
  };

  function formatEur(num) {
    return new Intl.NumberFormat("es-ES", {
      style: "currency",
      currency: "EUR",
      maximumFractionDigits: 0,
    }).format(num);
  }

  function round500(val) {
    return Math.round(val / 500) * 500;
  }

  function showError(message) {
    els.errorBanner.textContent = message;
    els.errorBanner.classList.remove("hidden");
  }

  function clearError() {
    els.errorBanner.textContent = "";
    els.errorBanner.classList.add("hidden");
  }

  function getByokState() {
    const rawKey = els.byokApiKey ? els.byokApiKey.value.trim() : "";
    if (!rawKey) {
      return { active: false, unrecognized: false, providerLabel: "" };
    }

    const lowered = rawKey.toLowerCase();
    const placeholders = new Set(["your_api_key", "your_key_here", "changeme", "sk-or-v1-", "sk-or-"]);
    if (
      placeholders.has(lowered) ||
      rawKey.includes("...") ||
      rawKey.includes("${") ||
      lowered.endsWith("xxx")
    ) {
      return { active: true, unrecognized: true, providerLabel: "" };
    }

    const selected = els.byokProvider ? els.byokProvider.value : "auto";
    const labels = {
      gemini: "Gemini",
      openrouter: "OpenRouter",
      anthropic: "Claude",
      openai: "OpenAI",
    };

    if (selected !== "auto" && labels[selected]) {
      return { active: true, unrecognized: false, providerLabel: labels[selected] };
    }

    if (rawKey.startsWith("AIza")) {
      return { active: true, unrecognized: false, providerLabel: "Gemini" };
    }
    if (rawKey.startsWith("sk-or-")) {
      return { active: true, unrecognized: false, providerLabel: "OpenRouter" };
    }
    if (rawKey.startsWith("sk-ant-")) {
      return { active: true, unrecognized: false, providerLabel: "Claude" };
    }
    if (rawKey.startsWith("sk-")) {
      return { active: true, unrecognized: false, providerLabel: "OpenAI" };
    }
    return { active: true, unrecognized: true, providerLabel: "" };
  }

  function renderVisionBadge(byok = getByokState()) {
    if (!els.visionBadge) return;

    els.visionBadge.replaceChildren();
    const dot = document.createElement("span");
    const label = document.createElement("span");

    if (byok.active && !byok.unrecognized) {
      els.visionBadge.className =
        "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-50 text-emerald-900 border border-emerald-200";
      dot.className = "h-2 w-2 rounded-full bg-emerald-500";
      label.textContent = `Tu API Key activa (${byok.providerLabel})`;
    } else if (byok.active && byok.unrecognized) {
      els.visionBadge.className =
        "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-amber-50 text-amber-900 border border-amber-200";
      dot.className = "h-2 w-2 rounded-full bg-amber-500";
      label.textContent = "Tu API Key (elige proveedor)";
    } else if (!state.serverHealthLoaded) {
      els.visionBadge.className =
        "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-[#F5F4EF] text-[#344E41] border border-[#DAD7CD]";
      dot.className = "h-2 w-2 rounded-full bg-slate-400";
      label.textContent = "Comprobando motor visual...";
    } else {
      const providerLabels = {
        openrouter: "OpenRouter",
        gemini: "Gemini API",
        openai: "OpenAI",
        anthropic: "Claude",
        demo: "Modo Demo Local",
      };
      const isLive = state.serverVisionMode !== "demo";
      const providerName = providerLabels[state.serverVisionMode] || state.serverVisionMode;
      els.visionBadge.className = isLive
        ? "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-50 text-emerald-900 border border-emerald-200"
        : "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-[#F5F4EF] text-[#344E41] border border-[#DAD7CD]";
      dot.className = `h-2 w-2 rounded-full ${isLive ? "bg-emerald-500" : "bg-amber-500"}`;
      label.textContent = isLive
        ? `${providerName} (${state.serverVisionModel})`
        : "Modo Demo Local (sin API key)";
    }

    els.visionBadge.appendChild(dot);
    els.visionBadge.appendChild(label);
  }

  function updateByokStatus() {
    const byok = getByokState();
    if (els.byokClearBtn) {
      els.byokClearBtn.disabled = !byok.active;
    }

    if (els.byokStatusPill) {
      if (byok.active && !byok.unrecognized) {
        els.byokStatusPill.className =
          "text-[11px] font-medium px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800";
        els.byokStatusPill.textContent = `Tu clave (${byok.providerLabel})`;
      } else if (byok.active && byok.unrecognized) {
        els.byokStatusPill.className =
          "text-[11px] font-medium px-2 py-0.5 rounded-full bg-amber-100 text-amber-800";
        els.byokStatusPill.textContent = "Elige proveedor";
      } else {
        const isDemo = state.serverVisionMode === "demo";
        els.byokStatusPill.className = isDemo
          ? "text-[11px] font-medium px-2 py-0.5 rounded-full bg-amber-100 text-amber-800"
          : "text-[11px] font-medium px-2 py-0.5 rounded-full bg-slate-200/80 text-slate-600";
        els.byokStatusPill.textContent = isDemo ? "Demo local" : "Motor servidor";
      }
    }

    renderVisionBadge(byok);
  }

  function renderToggles() {
    els.toggleBtns.forEach((btn) => {
      const key = btn.dataset.toggle;
      const active = Boolean(state.toggles[key]);
      btn.setAttribute("aria-pressed", String(active));
      const dot = btn.querySelector(".toggle-dot");

      if (active) {
        btn.classList.add("bg-[#344E41]", "text-white", "border-[#344E41]");
        btn.classList.remove("bg-white", "text-slate-600", "border-slate-300");
        dot.classList.add("bg-[#A3B18A]");
        dot.classList.remove("bg-slate-300");
      } else {
        btn.classList.remove("bg-[#344E41]", "text-white", "border-[#344E41]");
        btn.classList.add("bg-white", "text-slate-600", "border-slate-300");
        dot.classList.remove("bg-[#A3B18A]");
        dot.classList.add("bg-slate-300");
      }
    });
  }

  function populateMicroBarrios(preferredMicro = "") {
    const selectedDistrict = els.barrio.value;
    const distInfo = state.districtsData.get(selectedDistrict);
    const barrios = distInfo?.barrios || [];

    els.microBarrio.replaceChildren();
    const defaultOpt = document.createElement("option");
    defaultOpt.value = "";
    defaultOpt.textContent = barrios.length
      ? `Todo ${selectedDistrict} (media)`
      : "Media general del distrito";
    els.microBarrio.appendChild(defaultOpt);

    barrios.forEach((b) => {
      const opt = document.createElement("option");
      opt.value = b.nombre;
      opt.textContent = `${b.nombre} (~${b.mediana_m2.toLocaleString("es-ES")} €/m²)`;
      els.microBarrio.appendChild(opt);
    });

    if (preferredMicro && barrios.some((b) => b.nombre === preferredMicro)) {
      els.microBarrio.value = preferredMicro;
    } else {
      els.microBarrio.value = "";
    }
  }

  function updateDistrictHint() {
    const selected = els.barrio.value;
    const selectedMicro = els.microBarrio.value;
    const distInfo = state.districtsData.get(selected);
    if (!distInfo) {
      els.districtHint.textContent = "";
      return;
    }

    if (selectedMicro) {
      const match = (distInfo.barrios || []).find((b) => b.nombre === selectedMicro);
      if (match) {
        els.districtHint.textContent =
          `Mediana en ${selectedMicro}: ${match.mediana_m2.toLocaleString("es-ES")} €/m² ` +
          `(${match.muestras} testigos) · Distrito ${selected}: ${distInfo.mediana_m2.toLocaleString("es-ES")} €/m²`;
        return;
      }
    }
    els.districtHint.textContent = `Referencia mediana del distrito ${selected}: ${distInfo.mediana_m2.toLocaleString("es-ES")} €/m²`;
  }

  function renderPhotoPreviews() {
    els.photoCounter.textContent = `${state.photos.length} / ${state.maxFiles} fotos`;
    els.previewGrid.replaceChildren();

    if (state.photos.length === 0) {
      els.previewGrid.classList.add("hidden");
      return;
    }

    els.previewGrid.classList.remove("hidden");
    state.photos.forEach((item, index) => {
      const card = document.createElement("div");
      card.className = "relative group rounded-xl overflow-hidden border border-slate-200 aspect-square bg-slate-100";

      const img = document.createElement("img");
      img.src = item.url;
      img.alt = `Foto ${index + 1}`;
      img.className = "w-full h-full object-cover";

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.setAttribute("aria-label", `Eliminar foto ${index + 1}`);
      removeBtn.textContent = "×";
      removeBtn.className =
        "absolute top-1.5 right-1.5 h-6 w-6 rounded-full bg-slate-900/80 text-white text-sm leading-none flex items-center justify-center hover:bg-rose-600 transition";
      removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        URL.revokeObjectURL(item.url);
        state.photos.splice(index, 1);
        clearError();
        renderPhotoPreviews();
      });

      card.appendChild(img);
      card.appendChild(removeBtn);
      els.previewGrid.appendChild(card);
    });
  }

  function addFiles(fileList) {
    clearError();
    const incoming = Array.from(fileList || []);
    for (const file of incoming) {
      if (state.photos.length >= state.maxFiles) {
        showError(`Solo puedes adjuntar un máximo de ${state.maxFiles} fotografías por tasación.`);
        break;
      }
      if (!file.type.startsWith("image/")) {
        showError(`El archivo "${file.name}" no parece una imagen válida.`);
        continue;
      }
      if (file.size > state.maxSizeBytes) {
        showError(`La imagen "${file.name}" supera el límite de ${state.maxSizeMb} MB.`);
        continue;
      }
      state.photos.push({
        file,
        url: URL.createObjectURL(file),
      });
    }
    renderPhotoPreviews();
  }

  function applyPreset(name) {
    const preset = PRESETS[name];
    if (!preset) return;
    clearError();
    if (state.districtsData.has(preset.barrio)) {
      els.barrio.value = preset.barrio;
    }
    populateMicroBarrios(preset.micro_barrio);
    els.tipoVivienda.value = preset.tipo_vivienda;
    els.metros.value = preset.metros_cuadrados;
    els.habitaciones.value = preset.habitaciones;
    els.banos.value = preset.banos;
    els.planta.value = preset.planta;
    state.toggles.ascensor = preset.ascensor;
    state.toggles.exterior = preset.exterior;
    state.toggles.terraza = preset.terraza;
    state.toggles.garaje = preset.garaje;
    state.toggles.aire_acondicionado = preset.aire_acondicionado;
    state.toggles.trastero = preset.trastero;
    renderToggles();
    updateDistrictHint();
  }

  function renderTagList(container, items, emptyFallback, badgeColorClass) {
    container.replaceChildren();
    const entries = items && items.length > 0 ? items : [emptyFallback];
    entries.forEach((txt) => {
      const li = document.createElement("li");
      li.className = `px-3 py-2 rounded-lg border ${badgeColorClass}`;
      li.textContent = txt;
      container.appendChild(li);
    });
  }

  function renderTabularFactors(factors) {
    els.tabularFactorsList.replaceChildren();
    if (!factors || factors.length === 0) {
      els.tabularFactorsCard.classList.add("hidden");
      return;
    }
    els.tabularFactorsCard.classList.remove("hidden");

    factors.forEach((item) => {
      const positive = item.impacto_eur >= 0;
      const sign = positive ? "+" : "";
      const pill = document.createElement("span");
      pill.className = positive
        ? "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[#A3B18A] bg-[#F5F4EF] text-[#344E41] text-xs font-medium"
        : "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-rose-200 bg-rose-50 text-rose-900 text-xs font-medium";
      pill.textContent = `${item.factor}: ${sign}${formatEur(item.impacto_eur)} (${sign}${item.impacto_pct} %)`;
      els.tabularFactorsList.appendChild(pill);
    });
  }

  function updateSimulator() {
    const data = state.lastAppraisal;
    if (!data) return;

    const ref = Number(els.simReforma.value);
    const cal = Number(els.simCalidades.value);
    const lum = data.analisis_visual.luminosidad;

    els.simReformaLabel.textContent = `${ref} / 10`;
    els.simCalidadesLabel.textContent = `${cal} / 10`;

    let simulatedPrice;
    if (ref === data.analisis_visual.reforma && cal === data.analisis_visual.calidades) {
      simulatedPrice = data.valoracion.precio_ajustado;
    } else {
      const weighted = Math.round((0.45 * ref + 0.30 * cal + 0.25 * lum) * 100) / 100;
      const deltaNorm = Math.max(-1, Math.min(1, (weighted - 5.5) / 4.5));
      const adjPct = Math.round(deltaNorm * state.maxAdjustmentPct * 100) / 100;
      const baseUnrounded =
        data.valoracion.precio_ajustado / (1 + data.valoracion.ajuste_visual_pct / 100) ||
        data.valoracion.precio_base;
      simulatedPrice = Math.max(10000, round500(baseUnrounded * (1 + adjPct / 100)));
    }

    const simM2 = Math.round(simulatedPrice / data.propiedad.metros_cuadrados);
    const diffEur = simulatedPrice - data.valoracion.precio_ajustado;
    const diffPct = Math.round((diffEur / Math.max(data.valoracion.precio_ajustado, 1)) * 1000) / 10;

    els.simPrice.textContent = formatEur(simulatedPrice);
    els.simM2.textContent = `(${simM2.toLocaleString("es-ES")} €/m²)`;

    if (diffEur === 0) {
      els.simDelta.textContent = "0 € (estado actual)";
      els.simDelta.className = "text-sm font-bold text-slate-600";
    } else {
      const sign = diffEur > 0 ? "+" : "";
      els.simDelta.textContent = `${sign}${formatEur(diffEur)} (${sign}${diffPct} %)`;
      els.simDelta.className =
        diffEur > 0 ? "text-sm font-bold text-emerald-600" : "text-sm font-bold text-rose-600";
    }
  }

  function renderComparables(comparables) {
    els.comparablesBody.replaceChildren();
    if (!comparables || comparables.length === 0) {
      els.comparablesCard.classList.add("hidden");
      return;
    }
    els.comparablesCard.classList.remove("hidden");

    comparables.forEach((comp) => {
      const tr = document.createElement("tr");
      tr.className = "hover:bg-[#F5F4EF]/80";

      const cells = [
        { text: comp.micro_barrio || comp.distrito, cls: "py-2.5 pr-3 font-medium text-[#344E41]" },
        { text: comp.tipo_vivienda, cls: "py-2.5 pr-3" },
        { text: `${comp.metros_cuadrados}`, cls: "py-2.5 pr-3" },
        { text: `${comp.habitaciones}/${comp.banos}`, cls: "py-2.5 pr-3" },
        { text: `${comp.planta}`, cls: "py-2.5 pr-3" },
        { text: formatEur(comp.precio_estimado), cls: "py-2.5 pr-3 font-semibold text-[#344E41]" },
        { text: `${comp.precio_m2.toLocaleString("es-ES")} €`, cls: "py-2.5 pr-3" },
      ];

      cells.forEach((c) => {
        const td = document.createElement("td");
        td.className = c.cls;
        td.textContent = c.text;
        tr.appendChild(td);
      });

      const simTd = document.createElement("td");
      simTd.className = "py-2.5 text-right";
      const badge = document.createElement("span");
      badge.className =
        "inline-block px-2 py-0.5 rounded-full text-[11px] font-semibold bg-[#A3B18A]/25 text-[#344E41] border border-[#A3B18A]/70";
      badge.textContent = `${comp.similitud_pct} %`;
      simTd.appendChild(badge);
      tr.appendChild(simTd);

      els.comparablesBody.appendChild(tr);
    });
  }

  function loadHistory() {
    try {
      const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch (_err) {
      return [];
    }
  }

  function saveHistory() {
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(state.history));
    } catch (_err) {
      // Ignora cuotas de almacenamiento privadas
    }
  }

  function addToHistory(appraisal) {
    const p = appraisal.propiedad;
    const v = appraisal.valoracion;
    const fingerprint = JSON.stringify([
      p.barrio,
      p.micro_barrio || "",
      p.tipo_vivienda,
      p.metros_cuadrados,
      p.habitaciones,
      p.planta,
      v.precio_ajustado,
    ]);

    const filtered = state.history.filter((entry) => entry.fingerprint !== fingerprint);
    filtered.unshift({
      id: Date.now(),
      fingerprint,
      savedAt: new Date().toLocaleTimeString("es-ES", { hour: "2-digit", minute: "2-digit" }),
      appraisal,
    });
    state.history = filtered.slice(0, MAX_HISTORY_ITEMS);
    saveHistory();
    renderHistory();
  }

  function renderHistory() {
    els.historyList.replaceChildren();
    if (!state.history.length) {
      els.historyCard.classList.add("hidden");
      return;
    }
    els.historyCard.classList.remove("hidden");

    state.history.forEach((entry, idx) => {
      const p = entry.appraisal.propiedad;
      const v = entry.appraisal.valoracion;
      const row = document.createElement("div");
      row.className =
        "flex flex-wrap items-center justify-between gap-2 rounded-xl border border-[#DAD7CD] bg-[#F5F4EF]/80 px-3.5 py-2.5 text-xs";

      const info = document.createElement("div");
      const title = document.createElement("p");
      title.className = "font-semibold text-[#344E41]";
      const zone = p.micro_barrio ? `${p.micro_barrio} (${p.barrio})` : p.barrio;
      title.textContent = `${p.tipo_vivienda} · ${zone} · ${p.metros_cuadrados} m²`;

      const sub = document.createElement("p");
      sub.className = "text-slate-600 mt-0.5";
      sub.textContent =
        `${formatEur(v.precio_ajustado)} (${v.precio_m2_estimado.toLocaleString("es-ES")} €/m²) · ` +
        `Score ${v.score_visual_ponderado}/10 · ${entry.savedAt}`;

      info.appendChild(title);
      info.appendChild(sub);

      const actions = document.createElement("div");
      actions.className = "flex items-center gap-1.5";

      const loadBtn = document.createElement("button");
      loadBtn.type = "button";
      loadBtn.className =
        "px-2.5 py-1 rounded-lg bg-[#588157] hover:bg-[#3A5A40] text-white font-medium transition";
      loadBtn.textContent = "Ver";
      loadBtn.addEventListener("click", () => {
        renderAppraisalResult(entry.appraisal, { recordHistory: false });
        els.resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
      });

      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.setAttribute("aria-label", "Eliminar del historial");
      delBtn.className =
        "h-6 w-6 rounded-lg border border-slate-200 bg-white hover:bg-rose-50 hover:text-rose-600 text-slate-500 flex items-center justify-center transition";
      delBtn.textContent = "×";
      delBtn.addEventListener("click", () => {
        state.history.splice(idx, 1);
        saveHistory();
        renderHistory();
      });

      actions.appendChild(loadBtn);
      actions.appendChild(delBtn);

      row.appendChild(info);
      row.appendChild(actions);
      els.historyList.appendChild(row);
    });
  }

  function renderAppraisalResult(data, options = { recordHistory: true }) {
    state.lastAppraisal = data;
    const { propiedad: p, valoracion: v, analisis_visual: vis, fotos_analizadas: nFotos } = data;

    els.emptyState.classList.add("hidden");
    els.resultsPanel.classList.remove("hidden");

    // Banner modo demo / aviso de respaldo
    if (vis.warning || vis.mode === "demo") {
      els.warningBanner.classList.remove("hidden");
      const isLiveFailover = vis.mode !== "demo" && Boolean(vis.warning);
      els.warningBanner.className = isLiveFailover
        ? "rounded-xl border border-sky-200 bg-sky-50 text-sky-900 px-4 py-3 text-xs"
        : "rounded-xl border border-amber-200 bg-amber-50 text-amber-900 px-4 py-3 text-xs";
      els.warningBanner.textContent = vis.warning
        ? vis.warning
        : "Modo Demo Local activo: no se detectó ninguna clave API válida en .env, por lo que las puntuaciones visuales se calcularon con heurística local sobre las imágenes.";
    } else {
      els.warningBanner.classList.add("hidden");
    }

    const ascTxt = p.ascensor ? "con ascensor" : "sin ascensor";
    const ubicacionTxt = p.micro_barrio ? `${p.micro_barrio} (${p.barrio})` : p.barrio;
    els.propertySummary.textContent = `${p.tipo_vivienda} · ${ubicacionTxt} · ${p.metros_cuadrados} m² · ${p.habitaciones} hab · Planta ${p.planta} (${ascTxt})`;

    if (v.comparativa_barrio_pct !== null && v.comparativa_barrio_pct !== undefined && v.mediana_barrio_m2) {
      els.comparisonPill.classList.remove("hidden");
      const sign = v.comparativa_barrio_pct >= 0 ? "+" : "";
      const refName = p.micro_barrio || p.barrio;
      els.comparisonPill.textContent = `${sign}${v.comparativa_barrio_pct} % vs mediana ${refName} (${v.mediana_barrio_m2.toLocaleString("es-ES")} €/m²)`;
      els.comparisonPill.className =
        v.comparativa_barrio_pct >= 0
          ? "text-xs font-semibold px-2.5 py-1 rounded-full bg-[#588157]/45 text-[#DAD7CD] border border-[#A3B18A]/40"
          : "text-xs font-semibold px-2.5 py-1 rounded-full bg-amber-500/20 text-amber-200";
    } else {
      els.comparisonPill.classList.add("hidden");
    }

    els.resultRange.textContent = `${formatEur(v.rango_min)} – ${formatEur(v.rango_max)}`;
    els.resultCentral.textContent = formatEur(v.precio_ajustado);
    els.resultM2.textContent = `${v.precio_m2_estimado.toLocaleString("es-ES")} €/m²`;
    els.resultM2Range.textContent = `${v.precio_m2_rango_min.toLocaleString("es-ES")} – ${v.precio_m2_rango_max.toLocaleString("es-ES")} €/m²`;

    // Pasos
    els.stepBasePrice.textContent = formatEur(v.precio_base);
    const factorSuffix =
      v.factor_actualizacion_mercado && v.factor_actualizacion_mercado !== 1
        ? ` · mercado ×${v.factor_actualizacion_mercado}`
        : "";
    els.stepBaseM2.textContent = `${v.precio_m2_base.toLocaleString("es-ES")} €/m² base${factorSuffix}`;

    const pctSign = v.ajuste_visual_pct > 0 ? "+" : "";
    const eurSign = v.ajuste_visual_eur > 0 ? "+" : "";
    els.stepVisualPct.textContent = `${pctSign}${v.ajuste_visual_pct} %`;
    els.stepVisualPct.className =
      "text-lg font-bold mt-1 " +
      (v.ajuste_visual_pct > 0
        ? "text-emerald-600"
        : v.ajuste_visual_pct < 0
        ? "text-rose-600"
        : "text-slate-900");
    els.stepVisualEur.textContent = `${eurSign}${formatEur(v.ajuste_visual_eur)} sobre base`;

    els.stepBandPct.textContent = `±${v.banda_incertidumbre_pct} %`;
    els.stepPhotosCount.textContent = `${nFotos} foto(s) analizada(s)`;

    renderTabularFactors(v.factores_precio_base);

    const providerLabels = {
      openrouter: "OpenRouter",
      gemini: "Gemini API",
      openai: "OpenAI",
      anthropic: "Claude",
      demo: "Modo Demo Local",
    };

    // Visión
    els.visionModelCaption.textContent =
      vis.mode === "demo"
        ? `Motor local: ${vis.model_used}`
        : `${providerLabels[vis.mode] || vis.mode}: ${vis.model_used}`;
    els.visionEstadoPill.textContent = vis.estado_general;
    els.visionWeightedScore.textContent = `Score ${v.score_visual_ponderado} / 10`;

    els.scoreReformaVal.textContent = `${vis.reforma} / 10`;
    els.scoreCalidadesVal.textContent = `${vis.calidades} / 10`;
    els.scoreLuminosidadVal.textContent = `${vis.luminosidad} / 10`;

    els.barReforma.style.width = `${vis.reforma * 10}%`;
    els.barCalidades.style.width = `${vis.calidades * 10}%`;
    els.barLuminosidad.style.width = `${vis.luminosidad * 10}%`;

    els.visionJustification.textContent = vis.justificacion;

    renderTagList(
      els.strengthsList,
      vis.puntos_fuertes,
      "Sin puntos destacados adicionales.",
      "bg-[#F5F4EF] border-[#A3B18A]/70 text-[#344E41]"
    );
    renderTagList(
      els.improvementsList,
      vis.puntos_mejora,
      "Sin mejoras prioritarias detectadas.",
      "bg-amber-50 border-amber-200 text-amber-900"
    );

    // Inicializa simulador What-If
    els.simReforma.value = String(vis.reforma);
    els.simCalidades.value = String(vis.calidades);
    els.simLuminosidadNote.textContent =
      `Luminosidad fija según fotos: ${vis.luminosidad}/10 · Precio actual: ${formatEur(v.precio_ajustado)}.`;
    updateSimulator();

    // Testigos comparables reales
    renderComparables(data.comparables || []);

    if (options.recordHistory) {
      addToHistory(data);
    }
  }

  function renderModelModal() {
    const model = state.modelMeta;
    if (!model) return;

    els.modalDatasetSubtitle.textContent =
      model.dataset_year_note || `Fuente: ${model.dataset_source || "dataset tabular"}`;

    // KPIs
    els.modalKpis.replaceChildren();
    const kpiItems = [
      { label: "Viviendas reales", value: (model.total_rows || 0).toLocaleString("es-ES") },
      {
        label: "Cobertura",
        value: `${model.n_districts || 20} distritos · ${model.n_neighborhoods || 0} barrios`,
      },
      {
        label: "R² test / CV 5-Fold",
        value: `${model.metrics?.test_r2 ?? "-"} / ${model.metrics?.cv5_r2 ?? "-"}`,
      },
      {
        label: "MedAPE · Mercado",
        value: `${model.metrics?.test_medape_pct ?? "-"} % · ×${model.market_update_factor ?? 1}`,
      },
    ];
    kpiItems.forEach((item) => {
      const card = document.createElement("div");
      card.className = "rounded-xl border border-[#DAD7CD] bg-[#F5F4EF] p-3";
      const lbl = document.createElement("p");
      lbl.className = "text-[11px] font-medium text-slate-500";
      lbl.textContent = item.label;
      const val = document.createElement("p");
      val.className = "text-sm font-bold text-[#344E41] mt-0.5";
      val.textContent = item.value;
      card.appendChild(lbl);
      card.appendChild(val);
      els.modalKpis.appendChild(card);
    });

    // Benchmark table
    els.modalBenchmarkBody.replaceChildren();
    (model.benchmark || []).forEach((row) => {
      const tr = document.createElement("tr");
      const isWinner = String(row.modelo || "").includes("XGBoost");
      tr.className = isWinner ? "bg-[#A3B18A]/25 font-semibold text-[#344E41]" : "";

      const vals = [
        row.modelo,
        String(row.r2),
        `${row.mape_pct} %`,
        `${row.medape_pct} %`,
        formatEur(row.mae_eur),
      ];
      vals.forEach((v) => {
        const td = document.createElement("td");
        td.className = "py-2.5 px-3";
        td.textContent = v;
        tr.appendChild(td);
      });
      els.modalBenchmarkBody.appendChild(tr);
    });

    // Feature importances
    els.modalImportancesList.replaceChildren();
    (model.feature_importances || []).slice(0, 7).forEach((item) => {
      const wrapper = document.createElement("div");
      const header = document.createElement("div");
      header.className = "flex justify-between text-xs mb-1";
      const nameSpan = document.createElement("span");
      nameSpan.className = "text-slate-700 font-medium";
      nameSpan.textContent = item.grupo;
      const pctSpan = document.createElement("span");
      pctSpan.className = "font-semibold text-slate-900";
      pctSpan.textContent = `${item.importancia_pct} %`;
      header.appendChild(nameSpan);
      header.appendChild(pctSpan);

      const track = document.createElement("div");
      track.className = "h-2 rounded-full bg-[#DAD7CD]/45 overflow-hidden";
      const fill = document.createElement("div");
      fill.className = "h-full rounded-full bg-[#588157]";
      fill.style.width = `${Math.min(100, Math.max(2, item.importancia_pct))}%`;
      track.appendChild(fill);

      wrapper.appendChild(header);
      wrapper.appendChild(track);
      els.modalImportancesList.appendChild(wrapper);
    });
  }

  async function openModelModal() {
    if (!state.modelMeta) {
      try {
        const res = await fetch("/api/metadata");
        if (res.ok) {
          const meta = await res.json();
          state.modelMeta = meta.model || null;
        }
      } catch (_err) {
        // Muestra modal incluso si falla la recarga
      }
    }
    renderModelModal();
    els.modelModal.classList.remove("hidden");
  }

  function closeModelModal() {
    els.modelModal.classList.add("hidden");
  }

  async function downloadPdfReport() {
    if (!state.lastAppraisal) return;
    els.downloadPdfBtn.disabled = true;
    els.downloadPdfLabel.textContent = "Generando PDF...";

    try {
      const response = await fetch("/api/report/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(state.lastAppraisal),
      });
      if (!response.ok) {
        showError("No se pudo generar el informe PDF.");
        return;
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const p = state.lastAppraisal.propiedad;
      const zone = (p.micro_barrio || p.barrio).toLowerCase().replace(/[^a-z0-9]+/g, "_");
      a.href = url;
      a.download = `calcasa_${zone}_${Math.round(p.metros_cuadrados)}m2.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (_err) {
      showError("Error de red al descargar el informe PDF.");
    } finally {
      els.downloadPdfBtn.disabled = false;
      els.downloadPdfLabel.textContent = "Descargar informe PDF";
    }
  }

  async function initApp() {
    renderToggles();
    state.history = loadHistory();
    renderHistory();

    try {
      const [healthRes, metaRes] = await Promise.all([
        fetch("/api/health"),
        fetch("/api/metadata"),
      ]);

      if (healthRes.ok) {
        const health = await healthRes.json();
        state.serverVisionMode = health.vision_mode || "demo";
        state.serverVisionModel = health.vision_model || health.openrouter_model || "";
        state.serverHealthLoaded = true;
        const isLive = state.serverVisionMode !== "demo";

        if (els.byokHelperText) {
          els.byokHelperText.textContent = isLive
            ? "El servidor local ya tiene IA visual activa. Opcionalmente puedes introducir otra API key solo para esta tasación."
            : "La demo pública funciona gratis en Modo Demo Local sin usar claves del autor. Si quieres probar la auditoría visual IA real, pega aquí tu propia clave (por ejemplo de Google AI Studio u OpenRouter).";
        }
        updateByokStatus();
      }

      if (metaRes.ok) {
        const meta = await metaRes.json();
        state.modelMeta = meta.model || null;

        if (meta.limits?.max_images) {
          state.maxFiles = Number(meta.limits.max_images);
        }
        if (meta.limits?.max_image_size_mb) {
          state.maxSizeMb = Number(meta.limits.max_image_size_mb);
          state.maxSizeBytes = state.maxSizeMb * 1024 * 1024;
        }
        if (meta.limits?.max_visual_adjustment_pct !== undefined) {
          state.maxAdjustmentPct = Number(meta.limits.max_visual_adjustment_pct);
        }
        if (els.photoLimitLabel) {
          els.photoLimitLabel.textContent = `2. Fotografías (1 a ${state.maxFiles} imágenes)`;
        }
        els.dropZone.setAttribute(
          "aria-label",
          `Subir de 1 a ${state.maxFiles} fotografías de la vivienda`
        );
        renderPhotoPreviews();

        els.barrio.replaceChildren();
        (meta.districts || []).forEach((d) => {
          state.districtsData.set(d.nombre, {
            mediana_m2: d.mediana_m2,
            barrios: d.barrios || [],
          });
          const opt = document.createElement("option");
          opt.value = d.nombre;
          opt.textContent = `${d.nombre} (~${d.mediana_m2.toLocaleString("es-ES")} €/m²)`;
          els.barrio.appendChild(opt);
        });
        if (state.districtsData.has("Salamanca")) {
          els.barrio.value = "Salamanca";
          populateMicroBarrios("Recoletos");
        } else {
          populateMicroBarrios();
        }
        updateDistrictHint();

        const m = meta.model?.metrics;
        if (m) {
          const sourceLabel = meta.model.dataset_source || "dataset";
          els.metricsFooter.textContent =
            `Modelo tabular: XGBoost · R² test ${m.test_r2} · MAPE ${m.test_mape_pct}% · ` +
            `${meta.model.total_rows?.toLocaleString("es-ES")} viviendas (${sourceLabel})`;
        } else {
          els.metricsFooter.textContent = "Modelo tabular pendiente de entrenamiento.";
        }
      }
    } catch (_err) {
      showError("No se pudo conectar con la API local para cargar distritos y métricas.");
    }
  }

  // Event listeners
  els.barrio.addEventListener("change", () => {
    populateMicroBarrios();
    updateDistrictHint();
  });
  els.microBarrio.addEventListener("change", updateDistrictHint);

  els.toggleBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.toggle;
      state.toggles[key] = !state.toggles[key];
      renderToggles();
    });
  });

  els.presetBtns.forEach((btn) => {
    btn.addEventListener("click", () => applyPreset(btn.dataset.preset));
  });

  els.dropZone.addEventListener("click", () => els.photoInput.click());
  els.dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      els.photoInput.click();
    }
  });
  els.photoInput.addEventListener("change", (e) => {
    addFiles(e.target.files);
    els.photoInput.value = "";
  });

  els.dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    els.dropZone.classList.add("border-[#588157]", "bg-[#F5F4EF]");
  });
  els.dropZone.addEventListener("dragleave", () => {
    els.dropZone.classList.remove("border-[#588157]", "bg-[#F5F4EF]");
  });
  els.dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    els.dropZone.classList.remove("border-[#588157]", "bg-[#F5F4EF]");
    addFiles(e.dataTransfer.files);
  });

  els.simReforma.addEventListener("input", updateSimulator);
  els.simCalidades.addEventListener("input", updateSimulator);
  els.simBtnCurrent.addEventListener("click", () => {
    if (!state.lastAppraisal) return;
    els.simReforma.value = String(state.lastAppraisal.analisis_visual.reforma);
    els.simCalidades.value = String(state.lastAppraisal.analisis_visual.calidades);
    updateSimulator();
  });
  els.simBtnMid.addEventListener("click", () => {
    els.simReforma.value = "7";
    els.simCalidades.value = "7";
    updateSimulator();
  });
  els.simBtnFull.addEventListener("click", () => {
    els.simReforma.value = "9";
    els.simCalidades.value = "9";
    updateSimulator();
  });

  els.downloadPdfBtn.addEventListener("click", downloadPdfReport);

  els.clearHistoryBtn.addEventListener("click", () => {
    state.history = [];
    saveHistory();
    renderHistory();
  });

  els.openModelModalBtn.addEventListener("click", openModelModal);
  els.footerModelModalBtn.addEventListener("click", openModelModal);
  els.closeModelModalBtn.addEventListener("click", closeModelModal);
  els.modelModal.addEventListener("click", (e) => {
    if (e.target === els.modelModal) {
      closeModelModal();
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !els.modelModal.classList.contains("hidden")) {
      closeModelModal();
    }
  });

  els.form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError();

    if (state.photos.length < 1 || state.photos.length > state.maxFiles) {
      showError(
        `Debes subir entre 1 y ${state.maxFiles} fotografías de la vivienda para realizar la tasación.`
      );
      return;
    }

    const formData = new FormData();
    formData.append("barrio", els.barrio.value);
    if (els.microBarrio.value) {
      formData.append("micro_barrio", els.microBarrio.value);
    }
    formData.append("tipo_vivienda", els.tipoVivienda.value);
    formData.append("metros_cuadrados", els.metros.value);
    formData.append("habitaciones", els.habitaciones.value);
    formData.append("banos", els.banos.value);
    formData.append("planta", els.planta.value);
    formData.append("ascensor", String(state.toggles.ascensor));
    formData.append("exterior", String(state.toggles.exterior));
    formData.append("terraza", String(state.toggles.terraza));
    formData.append("garaje", String(state.toggles.garaje));
    formData.append("aire_acondicionado", String(state.toggles.aire_acondicionado));
    formData.append("trastero", String(state.toggles.trastero));
    state.photos.forEach((item) => {
      formData.append("fotos", item.file, item.file.name);
    });

    const byok = getByokState();
    els.submitBtn.disabled = true;
    els.submitLabel.textContent =
      byok.active && !byok.unrecognized
        ? `Analizando con ${byok.providerLabel}...`
        : "Analizando datos y fotografías...";

    try {
      const requestHeaders = {};
      const byokKey = els.byokApiKey ? els.byokApiKey.value.trim() : "";
      if (byokKey) {
        requestHeaders["X-Vision-Api-Key"] = byokKey;
        requestHeaders["X-Vision-Provider"] = els.byokProvider ? els.byokProvider.value || "auto" : "auto";
      }

      const response = await fetch("/api/appraise", {
        method: "POST",
        headers: requestHeaders,
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        const msg =
          typeof payload.detail === "string"
            ? payload.detail
            : "Revisa los campos introducidos o las imágenes adjuntas.";
        showError(msg);
        return;
      }
      renderAppraisalResult(payload, { recordHistory: true });
    } catch (_err) {
      showError("Error de red al comunicarse con el servidor de tasación.");
    } finally {
      els.submitBtn.disabled = false;
      els.submitLabel.textContent = "Calcular tasación multimodal";
    }
  });

  if (els.byokApiKey) {
    els.byokApiKey.addEventListener("input", updateByokStatus);
    els.byokApiKey.addEventListener("change", updateByokStatus);
  }
  if (els.byokProvider) {
    els.byokProvider.addEventListener("change", updateByokStatus);
  }
  if (els.byokToggleVis && els.byokApiKey) {
    els.byokToggleVis.addEventListener("click", () => {
      const showing = els.byokApiKey.type === "text";
      els.byokApiKey.type = showing ? "password" : "text";
      els.byokToggleVis.textContent = showing ? "Ver" : "Ocultar";
    });
  }
  if (els.byokClearBtn && els.byokApiKey) {
    els.byokClearBtn.addEventListener("click", () => {
      els.byokApiKey.value = "";
      els.byokApiKey.type = "password";
      if (els.byokToggleVis) {
        els.byokToggleVis.textContent = "Ver";
      }
      updateByokStatus();
    });
  }

  initApp();
})();
