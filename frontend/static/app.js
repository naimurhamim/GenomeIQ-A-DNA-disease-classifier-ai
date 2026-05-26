/* GenomeIQ — Alpine app logic */

const SAMPLES = {
  cancer:
    "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGT" +
    "GTCCCATCTGTCTGGAGTTGATCAAGGAACCTGTCTCCACAAAGTGTGACCACATATTTTGCAAATTTT" +
    "GCATGCTGAAACTTCTCAACCAGAAGAAAGGGCCTTCACAGTGTCCTTTATGTAAGAATGATATAACCA" +
    "AAAGGG",
  diabetes:
    "ATGGCCCTGTGGATGCGCCTCCTGCCCCTGCTGGCGCTGCTGGCCCTCTGGGGACCTGACCCAGCCGCAG" +
    "CCTTTGTGAACCAACACCTGTGCGGCTCACACCTGGTGGAAGCTCTCTACCTAGTGTGCGGGGAACGAG" +
    "GCTTCTTCTACACACCCAAGACCCGCCGGGAGGCAGAGGACCTGCAGGTGGGGCAGGTGGAGCTGGGCG" +
    "GTGGCCCTGGTGCAGGCAGCCTGCAGCCCTTGGCCCTGGAGGGGTCCCTGCAGAAGCGTGGCATTGTGG" +
    "AACAATGCTGT",
  alzheimers:
    "ATGCTGCCCGGTTTGGCACTGCTCCTGCTGGCCGCCTGGACGGCTCGGGCAGTGCAGAATTCTGACATG" +
    "CTGCAGAATTTCAGCCAGAATCAACCCGACTTCTCTGACTATGACAACAGCCACAGCAGCCAGCCAGAG" +
    "CCAGCCATGGAAGATGAGGATGAAGACGAAGATGAGGAAGATGAAGACGAGCCCAAAGAGGAAGATGAA" +
    "CCCAGGAAGGAGGAT",
  normal:
    "ATGGATGATGATATCGCCGCGCTCGTCGTCGACAACGGCTCCGGCATGTGCAAAGCCGGCTTCGCGGGC" +
    "GACGATGCCCCGAGGGCACTCTTCCAGCCTTCCTTCCTGGGCATGGAGTCCTGTGGCATCCACGAAACT" +
    "ACCTTCAACTCCATCATGAAGTGTGACGTGGACATCCGCAAAGACCTGTACGCCAACACACATGAAGAT" +
    "CAAGATCATTGCTCCTCCTGAGCGCAAGTACTCCGTGTGGATCGGCGGCTCCATCCTGGCCTCGCTGTC" +
    "CACCTTCCAG",
};

const DISEASE_THEME = {
  Cancer:     { color: "#ef4444", icon: "🔴", description: "Oncogenic gene markers detected." },
  Diabetes:   { color: "#f59e0b", icon: "🟠", description: "Insulin / glucose regulation markers detected." },
  Alzheimers: { color: "#a855f7", icon: "🟣", description: "Neurodegenerative pathway markers detected." },
  Normal:     { color: "#10b981", icon: "🟢", description: "No disease-associated markers detected." },
};

function genomeApp() {
  return {
    page: "analyze",
    theme: document.documentElement.classList.contains("dark") ? "dark" : "light",
    health: { ok: false, version: "" },
    knowledge: [],

    nav: [
      { id: "analyze",   icon: "🔬", label: "Analyze",          subtitle: "Predict disease class for a DNA sequence" },
      { id: "copilot",   icon: "💬", label: "Copilot",          subtitle: "Ask questions about your prediction & genomics" },
      { id: "mutation",  icon: "🧪", label: "Mutation Lab",     subtitle: "Compare reference vs variant impact" },
      { id: "batch",     icon: "📦", label: "Batch Upload",     subtitle: "Multi-sequence FASTA classification" },
      { id: "knowledge", icon: "📚", label: "Knowledge Base",   subtitle: "Disease information & gene markers" },
      { id: "manual",    icon: "📖", label: "User Manual",      subtitle: "How to use each feature, step by step" },
      { id: "slides",    icon: "🎤", label: "Presentation",     subtitle: "Slide deck for demos and reports" },
      { id: "about",     icon: "ℹ️",  label: "About",            subtitle: "Project details" },
    ],

    slide: 0,
    slideCount: 12,

    analyze: {
      sequence: "",
      loading: false,
      error: "",
      result: null,
      options: { explain: true, ood: true, similar: true, model: "tfidf" },
    },

    mutation: { reference: "", variant: "", loading: false, error: "", result: null },
    batch:    { loading: false, error: "", result: null },

    copilot: {
      messages: [],
      input: "",
      provider: "demo",
      providerStatus: { demo: { available: true } },
      kbFacts: 0,
      loading: false,
      error: "",
      suggestions: [
        "What is BRCA1 and how does it relate to cancer?",
        "Explain the role of APP and PSEN1 in Alzheimer's disease.",
        "What does the OOD detector mean by high-risk?",
        "Why might my Diabetes prediction have low confidence?",
        "How does GenomeIQ's classifier work?",
      ],
    },

    modelOptions: [
      { id: "tfidf",    label: "TF-IDF",    available: true,  description: "k-mer ensemble (fastest, default)" },
      { id: "dnabert2", label: "DNABERT-2", available: false, description: "Transformer (richer context)" },
      { id: "ensemble", label: "Ensemble",  available: false, description: "Average of both models" },
    ],

    liveStats: { length: "0", gc: "—", valid: false },
    _compChart: null,
    _statsTimer: null,

    /* ------------------------------------------------------------------ */

    async init() {
      await this.fetchHealth();
      await this.fetchKnowledge();
      await this.fetchModels();
      await this.fetchCopilotStatus();
      this.computeLiveStats();
      this.$watch("page", () => this.$nextTick(() => this.refreshAfterPageChange()));
      window.addEventListener("keydown", (e) => this.onSlideKey(e));
    },

    refreshAfterPageChange() {
      // ensure plots resize when navigating back
      if (this.page === "analyze" && this.analyze.result) {
        this.renderCompositionChart();
        this.renderCircularPlot();
      }
    },

    currentNav() {
      return this.nav.find((n) => n.id === this.page) || this.nav[0];
    },

    toggleTheme() {
      const isDark = document.documentElement.classList.toggle("dark");
      this.theme = isDark ? "dark" : "light";
      try { localStorage.setItem("genomeiq.theme", this.theme); } catch (_) {}
      // Re-render plots with theme-appropriate colors
      this.$nextTick(() => {
        if (this.analyze.result) {
          this.renderCompositionChart();
          this.renderCircularPlot();
        }
      });
    },

    async fetchHealth() {
      try {
        const r = await fetch("/health");
        const j = await r.json();
        this.health = { ok: r.ok, version: j.version || "" };
      } catch {
        this.health = { ok: false, version: "" };
      }
    },

    async fetchKnowledge() {
      try {
        const r = await fetch("/diseases");
        this.knowledge = await r.json();
      } catch {
        this.knowledge = [];
      }
    },

    async fetchModels() {
      try {
        const r = await fetch("/models");
        const j = await r.json();
        this.modelOptions = this.modelOptions.map((opt) => ({
          ...opt,
          available: !!j[opt.id]?.available,
          description: j[opt.id]?.description || opt.description,
        }));
        // If current selection unavailable, fall back to tfidf
        const current = this.modelOptions.find((o) => o.id === this.analyze.options.model);
        if (!current?.available) this.analyze.options.model = "tfidf";
      } catch {}
    },

    async fetchCopilotStatus() {
      try {
        const r = await fetch("/chat/status");
        const j = await r.json();
        this.copilot.providerStatus = j.providers || {};
        this.copilot.kbFacts = j.knowledge_base?.facts_total || 0;
        // If selected provider unavailable, fall back to demo
        if (!this.copilot.providerStatus[this.copilot.provider]?.available) {
          this.copilot.provider = "demo";
        }
      } catch {}
    },

    async sendCopilotMessage(text) {
      const message = (text ?? this.copilot.input ?? "").trim();
      if (!message || this.copilot.loading) return;

      this.copilot.error = "";
      this.copilot.messages.push({ role: "user", content: message });
      this.copilot.input = "";
      this.copilot.loading = true;

      const predictionContext = this.analyze.result
        ? {
            predicted_class: this.analyze.result.prediction.predicted_class,
            confidence: this.analyze.result.prediction.confidence,
            probabilities: this.analyze.result.prediction.probabilities,
            model: this.analyze.result.prediction.model,
          }
        : null;

      try {
        const r = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message,
            provider: this.copilot.provider,
            top_k: 5,
            history: this.copilot.messages.slice(-10).map((m) => ({ role: m.role, content: m.content })),
            prediction_context: predictionContext,
          }),
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || `Server error (${r.status})`);
        this.copilot.messages.push({
          role: "assistant",
          content: data.answer,
          retrieved: data.retrieved || [],
          provider: data.provider,
          note: data.note,
        });
        this.$nextTick(() => this.scrollCopilotToBottom());
      } catch (e) {
        this.copilot.error = e.message || String(e);
      } finally {
        this.copilot.loading = false;
      }
    },

    scrollCopilotToBottom() {
      const el = document.querySelector("#copilot-messages");
      if (el) el.scrollTop = el.scrollHeight;
    },

    clearCopilot() {
      this.copilot.messages = [];
      this.copilot.error = "";
    },

    formatMarkdown(text) {
      if (!text) return "";
      const escaped = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
      return escaped
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.+?)\*/g, "<em>$1</em>")
        .replace(/`([^`]+)`/g, '<code class="px-1 py-0.5 rounded bg-ink-100 dark:bg-ink-800 font-mono text-[12px]">$1</code>')
        .replace(/\n/g, "<br>");
    },

    /* ---- Presentation slide controls -------------------------------- */
    nextSlide() {
      if (this.slide < this.slideCount - 1) this.slide += 1;
    },
    prevSlide() {
      if (this.slide > 0) this.slide -= 1;
    },
    onSlideKey(e) {
      if (this.page !== "slides") return;
      if (e.key === "ArrowRight" || e.key === " ") this.nextSlide();
      else if (e.key === "ArrowLeft") this.prevSlide();
    },

    /* ---- Analyze page ----------------------------------------------- */

    onSequenceInput() {
      clearTimeout(this._statsTimer);
      this._statsTimer = setTimeout(() => this.computeLiveStats(), 80);
    },

    computeLiveStats() {
      const cleaned = (this.analyze.sequence || "").toUpperCase().replace(/[^ACGTN]/g, "");
      const n = cleaned.length;
      let g = 0, c = 0;
      for (let i = 0; i < n; i++) {
        const ch = cleaned[i];
        if (ch === "G") g++;
        else if (ch === "C") c++;
      }
      this.liveStats = {
        length: n.toLocaleString(),
        gc: n ? ((g + c) / n * 100).toFixed(1) + "%" : "—",
        valid: n >= 20,
      };
    },

    loadSample(key) {
      this.analyze.sequence = SAMPLES[key] || "";
      this.computeLiveStats();
    },

    async runAnalyze() {
      this.analyze.error = "";
      this.analyze.result = null;
      const seq = (this.analyze.sequence || "").trim();
      if (!seq) {
        this.analyze.error = "Please enter a DNA sequence.";
        return;
      }
      this.analyze.loading = true;
      try {
        const resp = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sequence: seq,
            include_explain: this.analyze.options.explain,
            include_ood: this.analyze.options.ood,
            include_similar: this.analyze.options.similar,
            top_k_similar: 5,
            model: this.analyze.options.model || "tfidf",
          }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || `Server error (${resp.status})`);
        this.analyze.result = data;
        this.$nextTick(() => {
          this.renderCompositionChart();
          this.renderCircularPlot();
        });
      } catch (e) {
        this.analyze.error = e.message || String(e);
      } finally {
        this.analyze.loading = false;
      }
    },

    sortedProbs() {
      const probs = this.analyze.result?.prediction?.probabilities || {};
      return Object.entries(probs).sort((a, b) => b[1] - a[1]);
    },

    diseaseColor(name) { return DISEASE_THEME[name]?.color || "#64748b"; },
    diseaseLabel(name) { return name; },

    verdictColor() {
      const cls = this.analyze.result?.prediction?.predicted_class;
      return DISEASE_THEME[cls]?.color || "#3b82f6";
    },
    verdictIcon() {
      const cls = this.analyze.result?.prediction?.predicted_class;
      return DISEASE_THEME[cls]?.icon || "🧬";
    },
    verdictDescription() {
      const cls = this.analyze.result?.prediction?.predicted_class;
      return DISEASE_THEME[cls]?.description || "";
    },

    oodClass() {
      const ood = this.analyze.result?.ood;
      if (!ood) return "";
      if (ood.risk === "low")    return "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-300";
      if (ood.risk === "medium") return "bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-300";
      return "bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-300";
    },
    oodIcon() {
      const ood = this.analyze.result?.ood;
      return ood?.risk === "low" ? "✅" : ood?.risk === "medium" ? "⚠️" : "🚨";
    },

    renderSaliency() {
      const exp = this.analyze.result?.explanation;
      if (!exp || !exp.scores?.length) return '<span class="text-ink-400">No saliency available.</span>';
      const seq = (this.analyze.sequence || "").toUpperCase();
      const scores = exp.scores;
      const blockSize = 60;
      let html = "";
      for (let i = 0; i < seq.length; i++) {
        if (i % blockSize === 0 && i !== 0) html += '\n';
        const s = scores[i] ?? 0;
        const bg = saliencyColor(s);
        const ch = seq[i] || "·";
        html += `<span style="background:${bg};padding:1px 1px;border-radius:2px;color:${s>0.5?'#0f172a':'#e2e8f0'}">${ch}</span>`;
      }
      return `<pre style="white-space:pre-wrap;word-break:break-all;margin:0">${html}</pre>`;
    },

    renderCompositionChart() {
      if (!this.$refs.compChart) return;
      const stats = this.analyze.result?.stats;
      if (!stats) return;
      if (this._compChart) this._compChart.destroy();
      const dark = document.documentElement.classList.contains("dark");
      const ctx = this.$refs.compChart.getContext("2d");
      const counts = stats.base_counts;
      this._compChart = new Chart(ctx, {
        type: "doughnut",
        data: {
          labels: ["A", "T", "G", "C", "N"],
          datasets: [{
            data: [counts.A, counts.T, counts.G, counts.C, counts.N],
            backgroundColor: ["#60a5fa", "#fb7185", "#34d399", "#fbbf24", "#94a3b8"],
            borderColor: dark ? "#0f172a" : "#ffffff",
            borderWidth: 2,
          }],
        },
        options: {
          plugins: {
            legend: {
              position: "bottom",
              labels: { font: { size: 11 }, color: dark ? "#cbd5e1" : "#334155" },
            },
          },
          cutout: "62%",
        },
      });
    },

    renderCircularPlot() {
      const node = this.$refs.circularPlot;
      if (!node || typeof Plotly === "undefined") return;
      const result = this.analyze.result;
      if (!result) return;
      const seq = (this.analyze.sequence || "").toUpperCase();
      const n = seq.length;
      if (!n) return;

      const dark = document.documentElement.classList.contains("dark");
      const traces = [];

      // GC profile (sliding-window) plotted as polar line on outer ring
      const win = Math.max(20, Math.floor(n / 200));
      const gcRadius = 1.0;
      const gcInner = 0.85;
      const gcVals = [];
      const gcAngles = [];
      for (let i = 0; i < n - win; i += Math.max(1, Math.floor(win / 2))) {
        let g = 0, c = 0;
        for (let j = i; j < i + win; j++) {
          const ch = seq[j];
          if (ch === "G") g++;
          else if (ch === "C") c++;
        }
        gcVals.push((g + c) / win);
        gcAngles.push((i / n) * 360);
      }
      const gcRads = gcVals.map((v) => gcInner + (gcRadius - gcInner) * v);
      traces.push({
        type: "scatterpolar",
        r: gcRads.concat([gcRads[0]]),
        theta: gcAngles.concat([gcAngles[0]]),
        mode: "lines",
        line: { color: "#3b82f6", width: 2 },
        name: "GC profile",
        hoverinfo: "skip",
      });

      // Saliency hotspots as colored markers on a middle ring
      const exp = result.explanation;
      if (exp && exp.scores && exp.scores.length) {
        const scores = exp.scores;
        const sampleStep = Math.max(1, Math.floor(scores.length / 720));
        const angles = [];
        const colors = [];
        const sizes = [];
        for (let i = 0; i < scores.length; i += sampleStep) {
          angles.push((i / n) * 360);
          colors.push(scores[i] || 0);
          sizes.push(4 + (scores[i] || 0) * 10);
        }
        traces.push({
          type: "scatterpolar",
          r: angles.map(() => 0.7),
          theta: angles,
          mode: "markers",
          marker: { color: colors, colorscale: [[0, "#1e293b"], [0.5, "#fbbf24"], [1, "#f43f5e"]], cmin: 0, cmax: 1, size: sizes, opacity: 0.8 },
          name: "Saliency",
          hoverinfo: "skip",
        });
      }

      // ORFs as colored arcs on inner ring
      const orfs = (result.orfs || []).slice(0, 12);
      orfs.forEach((orf, idx) => {
        const startAng = (orf.start / n) * 360;
        const endAng = (orf.end / n) * 360;
        const steps = Math.max(20, Math.floor((endAng - startAng) / 2));
        const thetas = [];
        for (let s = 0; s <= steps; s++) thetas.push(startAng + ((endAng - startAng) * s) / steps);
        const r = orf.strand === "+" ? 0.55 : 0.45;
        traces.push({
          type: "scatterpolar",
          r: thetas.map(() => r),
          theta: thetas,
          mode: "lines",
          line: { color: orf.strand === "+" ? "#10b981" : "#a855f7", width: 8 },
          name: `ORF ${idx + 1} (${orf.strand}${orf.frame})`,
          hovertext: thetas.map(() => `ORF ${orf.strand} frame ${orf.frame} · ${orf.protein_length} aa`),
          hoverinfo: "text",
          showlegend: false,
        });
      });

      const layout = {
        showlegend: false,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: { l: 0, r: 0, t: 0, b: 0 },
        polar: {
          bgcolor: "rgba(0,0,0,0)",
          radialaxis: { visible: false, range: [0, 1.1] },
          angularaxis: {
            tickmode: "array",
            tickvals: [0, 90, 180, 270],
            ticktext: [`0`, `${Math.round(n / 4)}`, `${Math.round(n / 2)}`, `${Math.round((3 * n) / 4)}`],
            color: dark ? "#94a3b8" : "#475569",
            gridcolor: dark ? "#334155" : "#e2e8f0",
            linecolor: dark ? "#334155" : "#cbd5e1",
            direction: "clockwise",
            rotation: 90,
          },
        },
        annotations: [
          {
            text: `<b>${result.prediction.predicted_class}</b><br><span style="font-size:11px">${(result.prediction.confidence * 100).toFixed(1)}%</span>`,
            font: { color: this.diseaseColor(result.prediction.predicted_class), size: 18 },
            showarrow: false,
            x: 0.5,
            y: 0.5,
            xref: "paper",
            yref: "paper",
          },
        ],
      };

      Plotly.react(node, traces, layout, { displayModeBar: false, responsive: true });
    },

    async downloadPdf() {
      const seq = (this.analyze.sequence || "").trim();
      if (!seq) return;
      try {
        const resp = await fetch("/report/pdf", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sequence: seq,
            include_explain: this.analyze.options.explain,
            include_ood: this.analyze.options.ood,
            include_similar: this.analyze.options.similar,
            top_k_similar: 5,
            model: this.analyze.options.model || "tfidf",
          }),
        });
        if (!resp.ok) {
          const j = await resp.json().catch(() => ({}));
          throw new Error(j.detail || "PDF generation failed.");
        }
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `genomeiq-report-${Date.now()}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      } catch (e) {
        this.analyze.error = e.message || String(e);
      }
    },

    /* ---- Mutation page ---------------------------------------------- */

    mutationSample() {
      this.mutation.reference = SAMPLES.cancer;
      this.mutation.variant = SAMPLES.cancer.substring(0, 50) + "C" + SAMPLES.cancer.substring(51);
    },

    async runMutation() {
      this.mutation.error = "";
      this.mutation.result = null;
      const ref = (this.mutation.reference || "").trim();
      const alt = (this.mutation.variant || "").trim();
      if (!ref || !alt) {
        this.mutation.error = "Both reference and variant sequences are required.";
        return;
      }
      this.mutation.loading = true;
      try {
        const r = await fetch("/mutation", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ reference: ref, variant: alt }),
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || `Server error (${r.status})`);
        this.mutation.result = data;
      } catch (e) {
        this.mutation.error = e.message || String(e);
      } finally {
        this.mutation.loading = false;
      }
    },

    effectClass(effect) {
      switch (effect) {
        case "synonymous": return "bg-emerald-100 dark:bg-emerald-500/20 text-emerald-700 dark:text-emerald-300";
        case "missense":   return "bg-amber-100 dark:bg-amber-500/20 text-amber-700 dark:text-amber-300";
        case "nonsense":   return "bg-rose-100 dark:bg-rose-500/20 text-rose-700 dark:text-rose-300";
        case "stop_loss":  return "bg-purple-100 dark:bg-purple-500/20 text-purple-700 dark:text-purple-300";
        default:           return "bg-ink-100 dark:bg-ink-700 text-ink-700 dark:text-ink-200";
      }
    },

    /* ---- Batch page ------------------------------------------------- */

    async onFile(event) {
      this.batch.error = "";
      this.batch.result = null;
      const file = event.target.files?.[0];
      if (!file) return;
      this.batch.loading = true;
      try {
        const fd = new FormData();
        fd.append("file", file);
        const r = await fetch("/batch", { method: "POST", body: fd });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || `Server error (${r.status})`);
        this.batch.result = data;
      } catch (e) {
        this.batch.error = e.message || String(e);
      } finally {
        this.batch.loading = false;
      }
    },
  };
}

function saliencyColor(s) {
  s = Math.max(0, Math.min(1, s));
  if (s < 0.5) {
    const t = s / 0.5;
    return blend([30, 41, 59], [251, 191, 36], t);
  }
  const t = (s - 0.5) / 0.5;
  return blend([251, 191, 36], [244, 63, 94], t);
}
function blend(a, b, t) {
  const r = Math.round(a[0] + (b[0] - a[0]) * t);
  const g = Math.round(a[1] + (b[1] - a[1]) * t);
  const bch = Math.round(a[2] + (b[2] - a[2]) * t);
  return `rgb(${r},${g},${bch})`;
}
