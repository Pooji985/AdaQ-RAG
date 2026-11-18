/**
 * AdaQ-RAG Research Workspace Client Controller
 * Intent-Aware Adaptive Retrieval-Augmented Generation
 */

document.addEventListener("DOMContentLoaded", () => {
  // ---------------------------------------------------------------------------
  // DOM Elements
  // ---------------------------------------------------------------------------
  const backendStatusText = document.getElementById("backend-status-text");
  const statusDot = document.getElementById("status-dot");

  // Section A: Pipeline Trace Nodes
  const nodeInput = document.getElementById("node-input");
  const nodeAnalyzer = document.getElementById("node-analyzer");
  const nodeRouter = document.getElementById("node-router");
  const nodeRetrieval = document.getElementById("node-retrieval");
  const nodeAssessment = document.getElementById("node-assessment");
  const nodeGenerator = document.getElementById("node-generator");

  const branchDense = document.getElementById("branch-dense");
  const branchHybrid = document.getElementById("branch-hybrid");
  const branchMultistep = document.getElementById("branch-multistep");

  const conn1 = document.getElementById("conn-1");
  const conn2 = document.getElementById("conn-2");
  const conn3 = document.getElementById("conn-3");
  const conn4 = document.getElementById("conn-4");
  const conn5 = document.getElementById("conn-5");

  const traceStrategyLabel = document.getElementById("trace-strategy-label");
  const traceAssessmentLabel = document.getElementById("trace-assessment-label");
  const pipelineStatusCaption = document.getElementById("pipeline-status-caption");

  // Section B: Example Cards
  const exampleHeaders = document.querySelectorAll(".example-card-header");
  const loadExampleBtns = document.querySelectorAll(".load-example-btn");

  // Section C: Workspace Controls
  const modeTabs = document.querySelectorAll(".mode-tab");
  const systemSelectionBlock = document.getElementById("system-selection-block");
  const comparisonWarningBanner = document.getElementById("comparison-warning-banner");
  const queryTextarea = document.getElementById("query-textarea");
  const topKSlider = document.getElementById("top-k-slider");
  const topKDisplay = document.getElementById("top-k-display");
  const btnQuickAnalyze = document.getElementById("btn-quick-analyze");
  const btnExecuteQuery = document.getElementById("btn-execute-query");
  const btnExecuteLabel = document.getElementById("btn-execute-label");
  const btnSpinner = document.getElementById("btn-spinner");
  const workspaceError = document.getElementById("workspace-error");
  const workspaceErrorMsg = document.getElementById("workspace-error-msg");

  // Section D: Results Views
  const idleWorkspaceState = document.getElementById("idle-workspace-state");
  const executionProgressState = document.getElementById("execution-progress-state");
  const progressStatusTitle = document.getElementById("progress-status-title");
  const progressStatusDesc = document.getElementById("progress-status-desc");
  const progressPerSystem = document.getElementById("progress-per-system");

  const resultsSingleView = document.getElementById("results-single-view");
  const singleSystemLabel = document.getElementById("single-system-label");
  const singleLatencyTag = document.getElementById("single-latency-tag");
  const singleCallsTag = document.getElementById("single-calls-tag");
  const singleTokensTag = document.getElementById("single-tokens-tag");
  const singleAnswerContent = document.getElementById("single-answer-content");

  const singleTraceStrategyBadge = document.getElementById("single-trace-strategy-badge");
  const singleTraceIntent = document.getElementById("single-trace-intent");
  const singleTraceComplexity = document.getElementById("single-trace-complexity");
  const singleTraceCandidateK = document.getElementById("single-trace-candidate-k");
  const singleTraceFinalK = document.getElementById("single-trace-final-k");
  const singleTraceReason = document.getElementById("single-trace-reason");
  const singleSubqueriesBox = document.getElementById("single-subqueries-box");
  const singleSubqueriesList = document.getElementById("single-subqueries-list");
  const singleEscalationNotice = document.getElementById("single-escalation-notice");

  const singleSourcesCount = document.getElementById("single-sources-count");
  const singleSourcesAccordion = document.getElementById("single-sources-accordion");

  // Section D: Comparative Mode Views
  const resultsCompareView = document.getElementById("results-compare-view");
  const compareQueryEcho = document.getElementById("compare-query-echo");

  // Section E: Live & Pilot Tables
  const subsectionLiveComparison = document.getElementById("subsection-live-comparison");
  const liveCompareTbody = document.getElementById("live-compare-tbody");
  const pilotTableTbody = document.getElementById("pilot-table-tbody");

  // ---------------------------------------------------------------------------
  // 1. Backend Connectivity & Health Check
  // ---------------------------------------------------------------------------
  async function checkBackendHealth() {
    try {
      const resp = await fetch("/health");
      if (resp.ok) {
        const data = await resp.json();
        backendStatusText.textContent = "Backend ready (AdaQ-RAG online)";
        statusDot.classList.remove("offline");
      } else {
        backendStatusText.textContent = "Service degraded";
        statusDot.classList.add("offline");
      }
    } catch {
      backendStatusText.textContent = "Offline (local server)";
      statusDot.classList.add("offline");
    }
  }

  // ---------------------------------------------------------------------------
  // 2. Section A: Pipeline Flow Trace Visualizer
  // ---------------------------------------------------------------------------
  function resetPipelineTrace() {
    [nodeInput, nodeAnalyzer, nodeRouter, nodeRetrieval, nodeAssessment, nodeGenerator].forEach(n => {
      n.classList.remove("active-trace", "bypass-trace");
    });
    [conn1, conn2, conn3, conn4, conn5].forEach(c => c.classList.remove("active-trace"));
    [branchDense, branchHybrid, branchMultistep].forEach(b => b.classList.remove("active-branch"));

    traceStrategyLabel.textContent = "Dynamic Routing Rules";
    traceAssessmentLabel.textContent = "Sufficiency Verification";
    pipelineStatusCaption.textContent = "Pipeline ready. Active execution path will be highlighted upon query submission.";
  }

  function updatePipelineTrace(system, routing, strategy) {
    resetPipelineTrace();
    const strat = (strategy || routing?.predicted_strategy || "DENSE").toUpperCase();

    nodeInput.classList.add("active-trace");
    conn1.classList.add("active-trace");

    if (system === "basic_rag") {
      // Baseline Dense without adaptive analysis
      nodeAnalyzer.classList.add("bypass-trace");
      nodeRouter.classList.add("bypass-trace");
      nodeAssessment.classList.add("bypass-trace");
      nodeRetrieval.classList.add("active-trace");
      branchDense.classList.add("active-branch");
      conn4.classList.add("active-trace");
      nodeGenerator.classList.add("active-trace");

      traceStrategyLabel.textContent = "Fixed Baseline";
      pipelineStatusCaption.textContent = "Executed Path: Basic RAG — Fixed dense vector search (bypassed analyzer, router, and assessment).";
      return;
    }

    if (system === "hybrid_rag") {
      // Baseline Hybrid without adaptive analysis
      nodeAnalyzer.classList.add("bypass-trace");
      nodeRouter.classList.add("bypass-trace");
      nodeAssessment.classList.add("bypass-trace");
      nodeRetrieval.classList.add("active-trace");
      branchHybrid.classList.add("active-branch");
      conn4.classList.add("active-trace");
      nodeGenerator.classList.add("active-trace");

      traceStrategyLabel.textContent = "Fixed Baseline";
      pipelineStatusCaption.textContent = "Executed Path: Hybrid RAG — Fixed dual-channel BM25 + FAISS with CrossEncoder reranking.";
      return;
    }

    // Adaptive RAG or Full AdaQ-RAG
    nodeAnalyzer.classList.add("active-trace");
    conn2.classList.add("active-trace");
    nodeRouter.classList.add("active-trace");
    conn3.classList.add("active-trace");
    nodeRetrieval.classList.add("active-trace");

    if (strat === "DENSE") {
      branchDense.classList.add("active-branch");
      traceStrategyLabel.textContent = "Selected: DENSE (Lightweight)";
    } else if (strat === "HYBRID") {
      branchHybrid.classList.add("active-branch");
      traceStrategyLabel.textContent = "Selected: HYBRID (Dual-Channel)";
    } else if (strat === "MULTI_STEP") {
      branchMultistep.classList.add("active-branch");
      traceStrategyLabel.textContent = "Selected: MULTI_STEP (Orchestrated)";
    }

    if (system === "full_adaq_rag" || strat === "MULTI_STEP") {
      nodeAssessment.classList.add("active-trace");
      conn4.classList.add("active-trace");
      traceAssessmentLabel.textContent = "Active Sufficiency Check";
    } else {
      nodeAssessment.classList.add("bypass-trace");
    }

    conn5.classList.add("active-trace");
    nodeGenerator.classList.add("active-trace");

    pipelineStatusCaption.textContent = `Executed Path: ${system === "full_adaq_rag" ? "Full AdaQ-RAG" : "Adaptive RAG"} — Intent: ${routing?.predicted_intent || "Detected"}, Strategy: ${strat}, Chunks: ${routing?.final_top_k || 5}.`;
  }

  // ---------------------------------------------------------------------------
  // 3. Section B: Expandable Benchmark Examples Accordion
  // ---------------------------------------------------------------------------
  exampleHeaders.forEach(header => {
    header.addEventListener("click", () => {
      const card = header.closest(".example-card");
      const body = card.querySelector(".example-card-body");
      const isExpanded = card.classList.contains("expanded");

      if (isExpanded) {
        card.classList.remove("expanded");
        body.classList.add("collapsed");
        header.setAttribute("aria-expanded", "false");
      } else {
        card.classList.add("expanded");
        body.classList.remove("collapsed");
        header.setAttribute("aria-expanded", "true");
      }
    });

    header.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        header.click();
      }
    });
  });

  loadExampleBtns.forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const qText = btn.getAttribute("data-query");
      const qSys = btn.getAttribute("data-sys") || "adaptive_rag";

      queryTextarea.value = qText;

      // Ensure Mode 1 is active
      const modeAdaptiveRadio = document.querySelector('input[name="exec-mode"][value="adaptive"]');
      if (modeAdaptiveRadio) {
        modeAdaptiveRadio.checked = true;
        setOperatingMode("adaptive");
      }

      // Set target pipeline
      const targetRadio = document.querySelector(`input[name="target-pipeline"][value="${qSys}"]`);
      if (targetRadio) {
        targetRadio.checked = true;
        document.querySelectorAll(".pill-radio").forEach(p => p.classList.remove("active"));
        targetRadio.closest(".pill-radio")?.classList.add("active");
      }

      // Smooth scroll to workspace
      document.getElementById("section-workspace")?.scrollIntoView({ behavior: "smooth" });
      queryTextarea.focus();
    });
  });

  // ---------------------------------------------------------------------------
  // 4. Section C: Mode Switching & Workspace Controls
  // ---------------------------------------------------------------------------
  function setOperatingMode(mode) {
    modeTabs.forEach(tab => tab.classList.remove("active"));
    const selectedTab = document.getElementById(`mode-tab-${mode}`);
    selectedTab?.classList.add("active");

    if (mode === "adaptive") {
      systemSelectionBlock.classList.remove("hidden");
      comparisonWarningBanner.classList.add("hidden");
      btnExecuteLabel.textContent = "Execute Adaptive RAG";
      btnQuickAnalyze.classList.remove("hidden");
    } else {
      systemSelectionBlock.classList.add("hidden");
      comparisonWarningBanner.classList.remove("hidden");
      btnExecuteLabel.textContent = "Compare All 4 Systems";
      btnQuickAnalyze.classList.add("hidden");
    }
  }

  modeTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const radio = tab.querySelector('input[type="radio"]');
      if (radio) {
        radio.checked = true;
        setOperatingMode(radio.value);
      }
    });
  });

  // Target pipeline pills in Mode 1
  document.querySelectorAll(".pill-radio").forEach(pill => {
    pill.addEventListener("click", () => {
      document.querySelectorAll(".pill-radio").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      const radio = pill.querySelector('input[type="radio"]');
      if (radio) {
        radio.checked = true;
        const sys = radio.value;
        if (sys === "full_adaq_rag") {
          btnExecuteLabel.textContent = "Execute Full AdaQ-RAG";
        } else if (sys === "adaptive_rag") {
          btnExecuteLabel.textContent = "Execute Adaptive RAG";
        } else if (sys === "hybrid_rag") {
          btnExecuteLabel.textContent = "Execute Hybrid Baseline";
        } else {
          btnExecuteLabel.textContent = "Execute Basic Baseline";
        }
      }
    });
  });

  topKSlider.addEventListener("input", (e) => {
    topKDisplay.textContent = e.target.value;
  });

  function showWorkspaceError(msg) {
    workspaceErrorMsg.textContent = msg;
    workspaceError.classList.remove("hidden");
  }

  function clearWorkspaceError() {
    workspaceError.classList.add("hidden");
    workspaceErrorMsg.textContent = "";
  }

  function getOperatingMode() {
    const checked = document.querySelector('input[name="exec-mode"]:checked');
    return checked ? checked.value : "adaptive";
  }

  function getTargetPipeline() {
    const checked = document.querySelector('input[name="target-pipeline"]:checked');
    return checked ? checked.value : "adaptive_rag";
  }

  // ---------------------------------------------------------------------------
  // 5. Action: Quick Routing Inspection (Offline Classifier & Rules)
  // ---------------------------------------------------------------------------
  btnQuickAnalyze.addEventListener("click", async () => {
    const query = queryTextarea.value.trim();
    if (!query) {
      showWorkspaceError("Please enter a question to analyze.");
      queryTextarea.focus();
      return;
    }

    clearWorkspaceError();
    btnQuickAnalyze.disabled = true;

    try {
      const resp = await fetch("/api/v1/rag/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: "Query analysis failed" }));
        throw new Error(err.detail || "Analysis request failed");
      }

      const data = await resp.json();

      // Show in single view
      idleWorkspaceState.classList.add("hidden");
      executionProgressState.classList.add("hidden");
      resultsCompareView.classList.add("hidden");
      resultsSingleView.classList.remove("hidden");

      singleSystemLabel.textContent = "ROUTING_ANALYSIS";
      singleLatencyTag.textContent = "Latency: < 50 ms";
      singleCallsTag.textContent = "Retrieval Calls: 0";
      singleTokensTag.textContent = "Tokens: None (No LLM generation)";

      singleAnswerContent.innerHTML = `<em>Deterministic query analysis completed without LLM generation. This inquiry is classified as <strong>${data.intent}</strong> with <strong>${data.complexity}</strong> complexity, routing to <strong>${data.strategy}</strong> retrieval with candidate pool top-${data.candidate_top_k} and final context top-${data.final_top_k}.<br><br>Click 'Execute Adaptive RAG' to retrieve supporting chunks and generate a grounded answer.</em>`;

      // Routing trace
      singleTraceStrategyBadge.textContent = `STRATEGY: ${data.strategy}`;
      singleTraceIntent.textContent = data.intent;
      singleTraceComplexity.textContent = data.complexity;
      singleTraceCandidateK.textContent = `Top-${data.candidate_top_k}`;
      singleTraceFinalK.textContent = `Top-${data.final_top_k}`;
      singleTraceReason.textContent = data.routing_reason;
      singleSubqueriesBox.classList.add("hidden");
      singleEscalationNotice.classList.add("hidden");

      // Sources
      singleSourcesCount.textContent = "0 Chunks";
      singleSourcesAccordion.innerHTML = `<div class="idle-text" style="padding: 12px;">No evidence chunks retrieved in analysis-only mode.</div>`;

      // Update flow trace
      updatePipelineTrace("adaptive_rag", {
        predicted_strategy: data.strategy,
        predicted_intent: data.intent,
        predicted_complexity: data.complexity,
        final_top_k: data.final_top_k,
      }, data.strategy);

      document.getElementById("section-answer-details")?.scrollIntoView({ behavior: "smooth" });

    } catch (err) {
      showWorkspaceError(`Analysis Error: ${err.message}`);
    } finally {
      btnQuickAnalyze.disabled = false;
    }
  });

  // ---------------------------------------------------------------------------
  // 6. Action: Execute Primary Pipeline (Mode 1: Single, Mode 2: Compare)
  // ---------------------------------------------------------------------------
  btnExecuteQuery.addEventListener("click", async () => {
    const query = queryTextarea.value.trim();
    if (!query) {
      showWorkspaceError("Please enter a question about scikit-learn.");
      queryTextarea.focus();
      return;
    }

    clearWorkspaceError();
    const mode = getOperatingMode();
    const topK = parseInt(topKSlider.value, 10) || 5;

    btnExecuteQuery.disabled = true;
    btnSpinner.classList.remove("hidden");

    if (mode === "adaptive") {
      await executeSingleSystem(query, getTargetPipeline(), topK);
    } else {
      await executeComparison(query, topK);
    }

    btnExecuteQuery.disabled = false;
    btnSpinner.classList.add("hidden");
  });

  // ---------------------------------------------------------------------------
  // 7. Single System Execution (Mode 1)
  // ---------------------------------------------------------------------------
  async function executeSingleSystem(query, system, topK) {
    idleWorkspaceState.classList.add("hidden");
    resultsCompareView.classList.add("hidden");
    resultsSingleView.classList.add("hidden");
    executionProgressState.classList.remove("hidden");
    progressPerSystem.classList.add("hidden");

    progressStatusTitle.textContent = `Executing ${system.toUpperCase()}...`;
    progressStatusDesc.textContent = "Analyzing query intent, gathering documentation evidence, and generating grounded response.";

    const startTime = performance.now();

    try {
      const resp = await fetch("/api/v1/rag/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          system,
          top_k: topK,
        }),
      });

      const elapsedMs = Math.round(performance.now() - startTime);

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: "Query execution failed" }));
        throw new Error(err.detail || `Server returned ${resp.status}`);
      }

      const data = await resp.json();

      executionProgressState.classList.add("hidden");
      resultsSingleView.classList.remove("hidden");

      renderSingleResult(data, elapsedMs);

      document.getElementById("section-answer-details")?.scrollIntoView({ behavior: "smooth" });

    } catch (err) {
      executionProgressState.classList.add("hidden");
      idleWorkspaceState.classList.remove("hidden");
      showWorkspaceError(`Pipeline Execution Error: ${err.message}`);
    }
  }

  function renderSingleResult(data, elapsedMs) {
    const sys = data.system || "adaptive_rag";
    singleSystemLabel.textContent = sys.toUpperCase();

    // Latency
    const latencyVal = data.latency_ms !== null && data.latency_ms !== undefined ? data.latency_ms : elapsedMs;
    singleLatencyTag.textContent = `Latency: ${(latencyVal / 1000).toFixed(1)}s (${latencyVal} ms)`;

    // Calls
    singleCallsTag.textContent = `Retrieval Calls: ${data.retrieval_calls !== null && data.retrieval_calls !== undefined ? data.retrieval_calls : 1}`;

    // Tokens
    if (data.metadata?.total_tokens) {
      singleTokensTag.textContent = `Tokens: ${data.metadata.total_tokens}`;
    } else {
      singleTokensTag.textContent = "Tokens: Not recorded";
    }

    // Grounded Answer
    singleAnswerContent.textContent = data.answer || "No response generated.";

    // Routing Information
    const routing = data.routing;
    if (routing) {
      singleTraceStrategyBadge.textContent = `STRATEGY: ${routing.predicted_strategy || "DENSE"}`;
      singleTraceIntent.textContent = routing.predicted_intent || "N/A";
      singleTraceComplexity.textContent = routing.predicted_complexity || "N/A";
      singleTraceCandidateK.textContent = routing.candidate_top_k ? `Top-${routing.candidate_top_k}` : "--";
      singleTraceFinalK.textContent = routing.final_top_k ? `Top-${routing.final_top_k}` : "--";
      singleTraceReason.textContent = routing.routing_reason || "Deterministic rule routing.";

      if (routing.sub_queries && routing.sub_queries.length > 0) {
        singleSubqueriesList.innerHTML = "";
        routing.sub_queries.forEach(sq => {
          const li = document.createElement("li");
          li.textContent = sq;
          singleSubqueriesList.appendChild(li);
        });
        singleSubqueriesBox.classList.remove("hidden");
      } else {
        singleSubqueriesBox.classList.add("hidden");
      }

      if (routing.was_escalated) {
        singleEscalationNotice.classList.remove("hidden");
      } else {
        singleEscalationNotice.classList.add("hidden");
      }
    } else {
      singleTraceStrategyBadge.textContent = "STRATEGY: DENSE";
      singleTraceIntent.textContent = "N/A (Fixed Baseline)";
      singleTraceComplexity.textContent = "N/A";
      singleTraceCandidateK.textContent = "--";
      singleTraceFinalK.textContent = `Top-${data.top_k || 5}`;
      singleTraceReason.textContent = "Fixed single-channel dense retrieval without query analysis.";
      singleSubqueriesBox.classList.add("hidden");
      singleEscalationNotice.classList.add("hidden");
    }

    // Supporting Sources
    const sources = data.sources || [];
    singleSourcesCount.textContent = `${sources.length} Chunks`;
    singleSourcesAccordion.innerHTML = "";

    if (sources.length === 0) {
      singleSourcesAccordion.innerHTML = `<div class="idle-text" style="padding: 12px;">No citation sources available for this query.</div>`;
    } else {
      sources.forEach((src, idx) => {
        const item = document.createElement("div");
        item.className = "source-item";

        const header = document.createElement("div");
        header.className = "source-item-header";
        header.innerHTML = `
          <div class="source-header-left">
            <span class="source-idx-tag">[Source ${src.source_index || idx + 1}]</span>
            <span class="source-title-text">${src.doc_title || "Scikit-Learn"} &bull; ${src.section_title || "General"}</span>
          </div>
          <span class="source-score-pill">Score: ${Number(src.score).toFixed(4)}</span>
        `;

        const body = document.createElement("div");
        body.className = "source-item-body collapsed";
        body.innerHTML = `
          <div class="source-meta-bar">
            Chunk ID: <code>${src.chunk_id}</code> | Section Level: ${src.section_level || 1}
          </div>
          <div class="source-snippet-text">${src.content_snippet || "(No content snippet available)"}</div>
        `;

        header.addEventListener("click", () => {
          body.classList.toggle("collapsed");
        });

        item.appendChild(header);
        item.appendChild(body);
        singleSourcesAccordion.appendChild(item);
      });
    }

    // Update Section A Flow Trace
    updatePipelineTrace(sys, routing, routing?.predicted_strategy);
  }

  // ---------------------------------------------------------------------------
  // 8. Comparative Mode Execution (Mode 2: 4 Systems)
  // ---------------------------------------------------------------------------
  async function executeComparison(query, topK) {
    idleWorkspaceState.classList.add("hidden");
    resultsSingleView.classList.add("hidden");
    executionProgressState.classList.remove("hidden");
    progressPerSystem.classList.remove("hidden");
    resultsCompareView.classList.add("hidden");

    progressStatusTitle.textContent = "Executing Comparative Evaluation...";
    progressStatusDesc.textContent = "Running query sequentially across Basic, Hybrid, Adaptive, and Full AdaQ-RAG with isolated error boundaries.";
    compareQueryEcho.textContent = `"${query}"`;

    // Reset status items
    const systemsList = [
      { key: "basic", name: "basic_rag" },
      { key: "hybrid", name: "hybrid_rag" },
      { key: "adaptive", name: "adaptive_rag" },
      { key: "full", name: "full_adaq_rag" },
    ];

    systemsList.forEach(({ key }) => {
      const item = document.getElementById(`prog-item-${key}`);
      if (item) {
        item.className = "system-status-item";
        item.querySelector(".prog-state").textContent = "Pending";
      }
      setCompareCardState(key, "pending", "Waiting for execution...");
    });

    try {
      // Execute via comparative endpoint
      const resp = await fetch("/api/v1/rag/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          top_k: topK,
        }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: "Comparison request failed" }));
        throw new Error(err.detail || `Server error (${resp.status})`);
      }

      const data = await resp.json();

      executionProgressState.classList.add("hidden");
      resultsCompareView.classList.remove("hidden");

      renderComparisonResults(query, data.systems);

      document.getElementById("section-answer-details")?.scrollIntoView({ behavior: "smooth" });

    } catch (err) {
      executionProgressState.classList.add("hidden");
      idleWorkspaceState.classList.remove("hidden");
      showWorkspaceError(`Comparison Error: ${err.message}`);
    }
  }

  function setCompareCardState(key, status, message) {
    const card = document.getElementById(`comp-card-${key}`);
    const badge = document.getElementById(`comp-status-${key}`);
    const ansText = document.getElementById(`comp-ans-${key}`);

    if (badge) {
      badge.className = `comp-status-badge status-${status}`;
      badge.textContent = status.toUpperCase();
    }
    if (ansText && message) {
      ansText.textContent = message;
    }
  }

  function renderComparisonResults(query, systemsData) {
    const mapping = [
      { key: "basic", sysName: "basic_rag", label: "Basic RAG (Baseline)" },
      { key: "hybrid", sysName: "hybrid_rag", label: "Hybrid RAG (Dual + Rerank)" },
      { key: "adaptive", sysName: "adaptive_rag", label: "Adaptive RAG (Dynamic)" },
      { key: "full", sysName: "full_adaq_rag", label: "Full AdaQ-RAG (Orchestrated)" },
    ];

    const liveSummaryRows = [];

    mapping.forEach(({ key, sysName, label }) => {
      const item = systemsData[sysName];
      const progItem = document.getElementById(`prog-item-${key}`);
      const retryBlock = document.getElementById(`comp-retry-${key}`);
      const ansElem = document.getElementById(`comp-ans-${key}`);
      const latElem = document.getElementById(`comp-lat-${key}`);
      const callsElem = document.getElementById(`comp-calls-${key}`);
      const routeElem = document.getElementById(`comp-route-${key}`);
      const srcCountElem = document.getElementById(`comp-src-count-${key}`);
      const srcContentElem = document.getElementById(`comp-sources-${key}`);

      if (!item) {
        setCompareCardState(key, "error", "No data returned.");
        return;
      }

      if (item.status === "success" && item.result) {
        const res = item.result;
        setCompareCardState(key, "success");
        if (progItem) {
          progItem.className = "system-status-item done";
          progItem.querySelector(".prog-state").textContent = "Success";
        }
        if (retryBlock) retryBlock.classList.add("hidden");

        ansElem.textContent = res.answer || "(Empty answer generated)";

        const latMs = res.latency_ms !== null && res.latency_ms !== undefined ? res.latency_ms : "--";
        const latSec = typeof latMs === "number" ? `${(latMs / 1000).toFixed(1)}s` : "--";
        latElem.textContent = `Latency: ${latSec}`;

        const calls = res.retrieval_calls !== null && res.retrieval_calls !== undefined ? res.retrieval_calls : "--";
        callsElem.textContent = `Calls: ${calls}`;

        const strat = res.routing?.predicted_strategy || (sysName === "hybrid_rag" ? "HYBRID" : "DENSE");
        routeElem.textContent = res.routing?.routing_reason || `Retrieved via ${strat} strategy.`;

        const sources = res.sources || [];
        srcCountElem.textContent = sources.length;
        srcContentElem.innerHTML = "";
        sources.forEach((s, idx) => {
          const cDiv = document.createElement("div");
          cDiv.className = "comp-source-chunk";
          cDiv.textContent = `[${idx + 1}] ${s.chunk_id} (score: ${Number(s.score).toFixed(3)})`;
          srcContentElem.appendChild(cDiv);
        });

        liveSummaryRows.push({
          system: label,
          strategy: strat,
          latency: latSec,
          calls: calls,
          sources: sources.length,
          status: "Success",
        });

      } else {
        // Error isolation
        setCompareCardState(key, "error", `Error: ${item.error || "Generation pipeline failed."}`);
        if (progItem) {
          progItem.className = "system-status-item error";
          progItem.querySelector(".prog-state").textContent = "Failed";
        }
        if (retryBlock) retryBlock.classList.remove("hidden");
        latElem.textContent = "Latency: Failed";
        callsElem.textContent = "Calls: --";

        liveSummaryRows.push({
          system: label,
          strategy: "N/A",
          latency: "--",
          calls: "--",
          sources: 0,
          status: "Failed",
        });
      }
    });

    // Wire up Sources view toggle
    document.querySelectorAll(".comp-sources-toggle .toggle-link").forEach(link => {
      link.onclick = () => {
        const targetId = link.getAttribute("data-target");
        document.getElementById(targetId)?.classList.toggle("collapsed");
      };
    });

    // Wire up single system retry buttons
    document.querySelectorAll(".retry-sys-btn").forEach(btn => {
      btn.onclick = async () => {
        const sysToRetry = btn.getAttribute("data-sys");
        await retrySingleSystemInCompare(query, sysToRetry);
      };
    });

    // Populate Section E Live Query Comparison Table
    renderLiveComparisonTable(liveSummaryRows);
  }

  async function retrySingleSystemInCompare(query, sysName) {
    const key = sysName.includes("basic") ? "basic" : sysName.includes("hybrid") ? "hybrid" : sysName.includes("full") ? "full" : "adaptive";
    setCompareCardState(key, "running", "Retrying execution...");
    const topK = parseInt(topKSlider.value, 10) || 5;

    try {
      const resp = await fetch("/api/v1/rag/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, system: sysName, top_k: topK }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: "Retry failed" }));
        throw new Error(err.detail || "Server error");
      }

      const res = await resp.json();
      const systemsUpdate = {};
      systemsUpdate[sysName] = { system: sysName, status: "success", result: res };
      renderComparisonResults(query, systemsUpdate);

    } catch (err) {
      setCompareCardState(key, "error", `Retry Failed: ${err.message}`);
    }
  }

  // ---------------------------------------------------------------------------
  // 9. Section E: Render Live & Pilot Benchmark Tables
  // ---------------------------------------------------------------------------
  function renderLiveComparisonTable(rows) {
    if (!rows || rows.length === 0) return;
    liveCompareTbody.innerHTML = "";
    rows.forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${r.system}</strong></td>
        <td><span class="font-mono">${r.strategy}</span></td>
        <td><span class="font-mono">${r.latency}</span></td>
        <td><span class="font-mono">${r.calls}</span></td>
        <td><span class="font-mono">${r.sources}</span></td>
        <td><span class="comp-status-badge status-${r.status.toLowerCase()}">${r.status}</span></td>
      `;
      liveCompareTbody.appendChild(tr);
    });
    subsectionLiveComparison.classList.remove("hidden");
  }

  async function loadPilotBenchmarkTable() {
    try {
      const resp = await fetch("/api/v1/evaluation/results");
      if (!resp.ok) return;
      const data = await resp.json();
      const comp = data.system_comparison;
      if (!comp || !pilotTableTbody) return;

      const order = [
        { key: "full_adaq_rag", label: "Full AdaQ-RAG", textClass: "system-text-full" },
        { key: "adaptive_rag", label: "Adaptive RAG", textClass: "system-text-adaptive" },
        { key: "hybrid_rag", label: "Hybrid RAG (Static)", textClass: "system-text-hybrid" },
        { key: "basic_rag", label: "Basic RAG (Baseline)", textClass: "system-text-basic" },
      ];

      pilotTableTbody.innerHTML = "";
      order.forEach(({ key, label, textClass }) => {
        const s = comp[key];
        if (!s) return;

        const quality = s.quality?.mean_overall?.toFixed(2) || "0.00";
        const recall = s.retrieval?.recall_at_5?.toFixed(4) || "0.0000";
        const mrr = s.retrieval?.mrr?.toFixed(4) || "0.0000";
        const routingAcc = (key === "adaptive_rag" || key === "full_adaq_rag")
          ? `<span class="accuracy-tag">${(s.routing?.strategy_accuracy * 100).toFixed(0)}%</span>`
          : "N/A (Fixed)";
        const meanCalls = s.efficiency?.mean_retrieval_calls?.toFixed(2) || "1.00";
        const meanLat = s.efficiency?.mean_latency_ms
          ? `${(s.efficiency.mean_latency_ms / 1000).toFixed(1)}s`
          : "--";

        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong class="font-mono ${textClass}">${label}</strong></td>
          <td><strong>${quality}</strong></td>
          <td>${recall}</td>
          <td>${mrr}</td>
          <td>${routingAcc}</td>
          <td>${meanCalls}</td>
          <td>${meanLat}</td>
        `;
        pilotTableTbody.appendChild(tr);
      });
    } catch {
      // Keep static table values if fetch fails
    }
  }

  // ---------------------------------------------------------------------------
  // 10. Initialization
  // ---------------------------------------------------------------------------
  checkBackendHealth();
  loadPilotBenchmarkTable();
});
