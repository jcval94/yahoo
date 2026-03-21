document.addEventListener('DOMContentLoaded', () => {
  const revealItems = document.querySelectorAll('.reveal');

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );

  revealItems.forEach((item) => observer.observe(item));

  const wireToggleTabs = ({ buttonSelector, panelSelector, targetAttr, panelAttr }) => {
    const buttons = document.querySelectorAll(buttonSelector);
    const panels = document.querySelectorAll(panelSelector);
    if (!buttons.length || !panels.length) return;

    buttons.forEach((button) => {
      button.addEventListener('click', () => {
        const target = button.getAttribute(targetAttr);
        buttons.forEach((item) => item.classList.toggle('is-active', item === button));
        panels.forEach((panel) => {
          const panelName = panel.getAttribute(panelAttr);
          panel.classList.toggle('is-hidden', panelName !== target);
        });
      });
    });
  };

  wireToggleTabs({
    buttonSelector: '.tab-button',
    panelSelector: '.tab-panel',
    targetAttr: 'data-tab-target',
    panelAttr: 'data-tab-panel',
  });

  wireToggleTabs({
    buttonSelector: '.subtab-button',
    panelSelector: '.subtab-panel',
    targetAttr: 'data-subtab-target',
    panelAttr: 'data-subtab-panel',
  });

  const analysisButtons = document.querySelectorAll('.analysis-selector-button');
  const analysisPanels = document.querySelectorAll('[data-analysis-panel]');
  analysisButtons.forEach((button) => {
    button.addEventListener('click', () => {
      const target = button.getAttribute('data-analysis-target');
      analysisButtons.forEach((b) => b.classList.toggle('is-active', b === button));
      analysisPanels.forEach((panel) => {
        const panelName = panel.getAttribute('data-analysis-panel');
        panel.classList.toggle('is-hidden', panelName !== target);
      });
    });
  });

  const images = document.querySelectorAll('img');
  images.forEach((image) => {
    image.addEventListener('error', () => {
      const source = image.getAttribute('src') || '';
      if (source.includes('?') && !image.dataset.retryWithoutQuery) {
        image.dataset.retryWithoutQuery = 'true';
        image.setAttribute('src', source.split('?')[0]);
        return;
      }
      image.classList.add('image-error');
      image.setAttribute('aria-label', 'No se pudo cargar la imagen');
      console.warn(`No se pudo cargar la imagen: ${image.getAttribute('src')}`);
    });
  });

  const parseCsv = (csvText) => {
    const lines = csvText.trim().split(/\r?\n/);
    if (lines.length < 2) return [];
    const headers = lines[0].split(',').map((h) => h.trim());
    return lines.slice(1).map((line) => {
      const cols = line.split(',');
      const row = {};
      headers.forEach((header, idx) => {
        row[header] = (cols[idx] || '').trim();
      });
      return row;
    });
  };

  const renderStrategyPerformance = (rows) => {
    const strategyTables = document.querySelectorAll('[data-strategy-performance-table] tbody');
    if (!strategyTables.length) return;

    const tableMarkup = rows.length
      ? rows
        .map((row) => `
          <tr>
            <td>${row.strategy}</td>
            <td>$${Number(row.ending_equity || 0).toFixed(2)}</td>
            <td>${Number(row.return_pct || 0).toFixed(2)}%</td>
            <td>${Number(row.win_rate || 0).toFixed(2)}%</td>
            <td>${Number(row.max_drawdown || 0).toFixed(4)}</td>
            <td>$${Number(row.initial_budget || 0).toFixed(2)}</td>
            <td>${Number(row.max_position_pct || 0).toFixed(2)}%</td>
            <td>$${Number(row.min_trade_usd || 0).toFixed(2)}</td>
            <td>${Number(row.holding_days || 0)}</td>
          </tr>
        `)
        .join('')
      : '<tr><td colspan="9">No hay datos disponibles.</td></tr>';

    strategyTables.forEach((tableBody) => {
      tableBody.innerHTML = tableMarkup;
    });
  };

  const renderStrategyBudgetAction = (rows) => {
    const tableBody = document.querySelector('#strategy-budget-action-table tbody');
    if (!tableBody) return;

    if (!rows.length) {
      tableBody.innerHTML = '<tr><td colspan="9">No hay datos disponibles.</td></tr>';
      return;
    }

    tableBody.innerHTML = rows.map((row) => `
      <tr>
        <td>${row.strategy || '-'}</td>
        <td>$${Number(row.initial_budget || 0).toFixed(2)}</td>
        <td>$${Number(row.ending_equity || 0).toFixed(2)}</td>
        <td>${Number(row.return_pct || 0).toFixed(2)}%</td>
        <td><span class="action-badge action-${String(row.latest_action || 'HOLD').toLowerCase()}">${row.latest_action || 'HOLD'}</span></td>
        <td>${row.latest_action_date || 'n/a'}</td>
        <td>${Number(row.buy_count || 0)}</td>
        <td>${Number(row.sell_count || 0)}</td>
        <td>${Number(row.hold_count || 0)}</td>
      </tr>
    `).join('');
  };

  const renderStrategyActionHistory = (rows) => {
    const tableBody = document.querySelector('#strategy-action-history-table tbody');
    if (!tableBody) return;

    if (!rows.length) {
      tableBody.innerHTML = '<tr><td colspan="6">No hay datos disponibles.</td></tr>';
      return;
    }

    tableBody.innerHTML = rows.map((row) => `
      <tr>
        <td>${row.date || 'n/a'}</td>
        <td>${Number(row.buy_count || 0)}</td>
        <td>${Number(row.sell_count || 0)}</td>
        <td>${Number(row.hold_count || 0)}</td>
        <td>${Number(row.tickers || 0)}</td>
        <td>${Number(row.avg_strategy_score || 0).toFixed(4)}</td>
      </tr>
    `).join('');
  };

  const renderPipelineHealth = (rows) => {
    const row = rows[0] || null;
    const runDate = document.getElementById('health-run-date');
    const duration = document.getElementById('health-duration');
    const success = document.getElementById('health-success');
    const fallback = document.getElementById('health-fallback');

    if (!runDate || !duration || !success || !fallback) return;
    if (!row) {
      runDate.textContent = 'Sin datos';
      duration.textContent = 'N/D';
      success.textContent = '0%';
      fallback.textContent = 'Sin datos';
      return;
    }

    runDate.textContent = row.run_date || 'N/D';
    duration.textContent = row.duration_minutes && row.duration_minutes !== 'n/a'
      ? `${Number(row.duration_minutes).toFixed(2)} min`
      : 'N/D';
    success.textContent = `${Number(row.success_pct || 0).toFixed(2)}% (${row.successful_steps || 0}/${row.total_steps || 0})`;
    fallback.textContent = row.fallback_offline || 'No detectado';
  };

  const renderActionRecommendations = (rows) => {
    const actionTable = document.getElementById('action-recommendations-table');
    const tableHeadRow = actionTable?.querySelector('thead tr');
    const tableBody = actionTable?.querySelector('tbody');
    if (!actionTable || !tableHeadRow || !tableBody) return;

    const staticColumns = [
      { key: 'date', label: 'Fecha', type: 'text' },
      { key: 'ticker', label: 'Ticker', type: 'text' },
      { key: 'best_model', label: 'Modelo líder', type: 'text' },
      { key: 'strategy_score', label: 'Score', type: 'number', formatter: (value) => Number(value || 0).toFixed(4) },
      { key: 'action', label: 'Acción', type: 'text' },
      { key: 'ret_5d', label: 'Resultado 5d', type: 'number' },
      { key: 'result_5d', label: 'Calidad 5d', type: 'text' },
    ];

    const normalizeModelToken = (value) => String(value || '').toLowerCase().replace(/[^a-z0-9]/g, '');

    const canonicalModelLabel = (columnName) => {
      const rawModel = columnName.replace('pred_', '');
      const normalized = normalizeModelToken(rawModel);
      const aliases = {
        rf: 'RandomForestRegressor',
        randomforest: 'RandomForestRegressor',
        randomforestregressor: 'RandomForestRegressor',
        ridge: 'Ridge',
        linreg: 'Ridge',
        linearregression: 'Ridge',
        xgb: 'XGBRegressor',
        xgbregressor: 'XGBRegressor',
        lgbm: 'LGBMRegressor',
        lgbmregressor: 'LGBMRegressor',
        arima: 'ARIMAModel',
        arimamodel: 'ARIMAModel',
        lstm: 'LSTM',
        lstmmodel: 'LSTM',
      };
      return aliases[normalized] || rawModel;
    };

    const canonicalModelName = (columnName) => normalizeModelToken(canonicalModelLabel(columnName));

    const scoreModelColumn = (columnName, preferredLabel = '') => {
      const rawModel = columnName.replace('pred_', '');
      const preferredCol = preferredLabel ? `pred_${preferredLabel}` : '';
      if (preferredCol && columnName === preferredCol) return 100;
      const normalizedRaw = normalizeModelToken(rawModel);
      const normalizedPreferred = normalizeModelToken(preferredLabel);
      if (normalizedPreferred && normalizedRaw === normalizedPreferred) return 90;
      let score = 0;
      if (/^[a-z0-9_]+$/.test(rawModel)) score += 2;
      if (/(model|regressor)/i.test(rawModel)) score += 1;
      score += rawModel.length / 100;
      return score;
    };

    const modelGroups = Array.from(
      rows.reduce((allColumns, row) => {
        Object.keys(row || {}).forEach((key) => {
          if (key.startsWith('pred_')) {
            allColumns.add(key);
          }
        });
        return allColumns;
      }, new Set())
    )
      .sort((a, b) => a.localeCompare(b))
      .reduce((groups, columnName) => {
        const canonicalName = canonicalModelName(columnName);
        const existing = groups.find((entry) => entry.canonicalName === canonicalName);
        if (!existing) {
          groups.push({
            canonicalName,
            labelColumn: columnName,
            displayLabel: canonicalModelLabel(columnName),
            sourceColumns: [columnName],
          });
          return groups;
        }

        existing.sourceColumns.push(columnName);
        if (scoreModelColumn(columnName, existing.displayLabel) > scoreModelColumn(existing.labelColumn, existing.displayLabel)) {
          existing.labelColumn = columnName;
        }
        return groups;
      }, []);

    const allColumns = [
      ...staticColumns,
      ...modelGroups.map((group) => ({
        key: group.canonicalName,
        label: group.displayLabel.toUpperCase(),
        type: 'number',
      })),
    ];

    const controls = document.getElementById('action-table-controls');
    const state = {
      sortKey: 'strategy_score',
      sortDirection: 'desc',
      filters: {},
      page: 1,
      pageSize: 20,
    };

    const getColumnValue = (row, key) => {
      const modelGroup = modelGroups.find((group) => group.canonicalName === key);
      if (!modelGroup) return row[key];

      const preferredSourceColumns = [...modelGroup.sourceColumns]
        .sort((a, b) => scoreModelColumn(b, modelGroup.displayLabel) - scoreModelColumn(a, modelGroup.displayLabel));
      for (const sourceKey of preferredSourceColumns) {
        const value = row[sourceKey];
        if (value !== undefined && value !== null && String(value).trim() !== '') {
          return value;
        }
      }
      return '';
    };

    const picklistFilters = [
      { key: 'date', label: 'Fecha' },
      { key: 'ticker', label: 'Ticker' },
      { key: 'best_model', label: 'Modelo líder' },
      { key: 'action', label: 'Acción' },
    ];

    const getFilterOptions = (key) => Array.from(new Set(
      rows
        .map((row) => String(getColumnValue(row, key) || '').trim())
        .filter(Boolean)
    )).sort((a, b) => a.localeCompare(b));

    const parseComparable = (value, type) => {
      if (value === null || value === undefined || value === '') return null;
      if (type === 'number') {
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : null;
      }
      return String(value).toLowerCase();
    };

    const renderRows = () => {
      const filtered = rows.filter((row) => allColumns.every((column) => {
        const filterValue = (state.filters[column.key] || '').toString().trim().toLowerCase();
        if (!filterValue) return true;
        return String(getColumnValue(row, column.key) || '').trim().toLowerCase() === filterValue;
      }));

      const sorted = [...filtered].sort((left, right) => {
        if (!state.sortKey) return 0;
        const column = allColumns.find((col) => col.key === state.sortKey);
        const leftValue = parseComparable(getColumnValue(left, state.sortKey), column?.type || 'text');
        const rightValue = parseComparable(getColumnValue(right, state.sortKey), column?.type || 'text');
        if (leftValue === rightValue) {
          if (state.sortKey === 'date') {
            const leftScore = parseComparable(left.strategy_score, 'number');
            const rightScore = parseComparable(right.strategy_score, 'number');
            if (leftScore === rightScore) return 0;
            if (leftScore === null) return 1;
            if (rightScore === null) return -1;
            return rightScore - leftScore;
          }
          return 0;
        }
        if (leftValue === null) return 1;
        if (rightValue === null) return -1;
        const factor = state.sortDirection === 'asc' ? 1 : -1;
        return leftValue > rightValue ? factor : -factor;
      });

      const totalRows = sorted.length;
      const showingAll = state.pageSize === 'all';
      const totalPages = showingAll ? 1 : Math.max(1, Math.ceil(totalRows / state.pageSize));
      state.page = Math.min(state.page, totalPages);
      const startIndex = showingAll ? 0 : (state.page - 1) * state.pageSize;
      const paginated = showingAll
        ? sorted
        : sorted.slice(startIndex, startIndex + state.pageSize);

      const summaryNode = controls?.querySelector('#action-page-summary');
      const prevButton = controls?.querySelector('#action-page-prev');
      const nextButton = controls?.querySelector('#action-page-next');
      if (summaryNode) {
        if (!totalRows) {
          summaryNode.textContent = 'Mostrando 0–0 de 0';
        } else {
          const from = startIndex + 1;
          const to = startIndex + paginated.length;
          summaryNode.textContent = `Mostrando ${from}–${to} de ${totalRows}`;
        }
      }
      if (prevButton) prevButton.disabled = state.page <= 1;
      if (nextButton) nextButton.disabled = state.page >= totalPages;

      if (!sorted.length) {
        const totalColumns = staticColumns.length + (modelGroups.length || 1);
        tableBody.innerHTML = `<tr><td colspan="${totalColumns}">No hay filas para los filtros seleccionados.</td></tr>`;
        return;
      }

      tableBody.innerHTML = paginated
        .map((row) => {
          const action = (row.action || 'HOLD').toUpperCase();
          const cls = action === 'BUY' ? 'action-buy' : action === 'SELL' ? 'action-sell' : 'action-hold';

          const staticCells = staticColumns.map((column) => {
            if (column.key === 'action') {
              return `<td><span class="action-badge ${cls}">${action}</span></td>`;
            }
            const rawValue = row[column.key];
            const value = rawValue === undefined || rawValue === '' ? '-' : rawValue;
            return `<td>${column.formatter ? column.formatter(value) : value}</td>`;
          });

          const modelCells = modelGroups.map((group) => {
            const value = getColumnValue(row, group.canonicalName);
            return `<td class="model-predictions">${value || '-'}</td>`;
          });

          return `
            <tr>
              ${staticCells.join('')}
              ${modelCells.join('')}
            </tr>
          `;
        })
        .join('');
    };

    const renderTableHead = () => {
      tableHeadRow.innerHTML = [
        ...staticColumns.map((column) => {
          const isActive = state.sortKey === column.key;
          const arrow = isActive ? (state.sortDirection === 'asc' ? ' ↑' : ' ↓') : '';
          const sortAttr = isActive
            ? (state.sortDirection === 'asc' ? 'ascending' : 'descending')
            : 'none';
          return `<th class="sortable-col" data-sort-key="${column.key}" aria-sort="${sortAttr}">${column.label}${arrow}</th>`;
        }),
        ...modelGroups.map((group) => {
          const isActive = state.sortKey === group.canonicalName;
          const arrow = isActive ? (state.sortDirection === 'asc' ? ' ↑' : ' ↓') : '';
          const sortAttr = isActive
            ? (state.sortDirection === 'asc' ? 'ascending' : 'descending')
            : 'none';
          return `<th class="model-prediction-col sortable-col" data-sort-key="${group.canonicalName}" aria-sort="${sortAttr}">${group.displayLabel.toUpperCase()}${arrow}</th>`;
        }),
      ].join('');

      tableHeadRow.querySelectorAll('[data-sort-key]').forEach((headerCell) => {
        headerCell.addEventListener('click', () => {
          const sortKey = headerCell.getAttribute('data-sort-key');
          if (!sortKey) return;
          if (state.sortKey === sortKey) {
            state.sortDirection = state.sortDirection === 'desc' ? 'asc' : 'desc';
          } else {
            state.sortKey = sortKey;
            state.sortDirection = 'desc';
          }
          state.page = 1;
          renderTableHead();
          renderRows();
        });
      });
    };

    if (controls) {
      controls.innerHTML = `
        ${picklistFilters.map((column) => `
          <label>
            Filtrar ${column.label}
            <select id="filter-${column.key}">
              <option value="">Todos</option>
              ${getFilterOptions(column.key).map((option) => `<option value="${option}">${option}</option>`).join('')}
            </select>
          </label>
        `).join('')}
        <label>
          Tamaño de página
          <select id="action-page-size">
            <option value="20">20</option>
            <option value="50">50</option>
            <option value="100">100</option>
          </select>
        </label>
        <button type="button" id="action-show-all">Ver todo</button>
        <button type="button" id="action-page-prev">Anterior</button>
        <button type="button" id="action-page-next">Siguiente</button>
        <span id="action-page-summary">Mostrando 0–0 de 0</span>
      `;

      const pageSizeSelect = controls.querySelector('#action-page-size');
      const showAllButton = controls.querySelector('#action-show-all');
      const prevButton = controls.querySelector('#action-page-prev');
      const nextButton = controls.querySelector('#action-page-next');

      if (pageSizeSelect) {
        pageSizeSelect.value = String(state.pageSize);
      }
      pageSizeSelect?.addEventListener('change', (event) => {
        state.pageSize = Number(event.target.value) || 20;
        state.page = 1;
        renderRows();
      });

      showAllButton?.addEventListener('click', () => {
        state.pageSize = 'all';
        state.page = 1;
        if (pageSizeSelect) pageSizeSelect.value = '20';
        renderRows();
      });

      prevButton?.addEventListener('click', () => {
        if (state.page <= 1) return;
        state.page -= 1;
        renderRows();
      });

      nextButton?.addEventListener('click', () => {
        state.page += 1;
        renderRows();
      });

      picklistFilters.forEach((column) => {
        const input = controls.querySelector(`#filter-${column.key}`);
        input?.addEventListener('change', (event) => {
          state.filters[column.key] = event.target.value;
          state.page = 1;
          renderRows();
        });
      });
    }

    renderTableHead();

    if (!rows.length) {
      const totalColumns = staticColumns.length + (modelGroups.length || 1);
      tableBody.innerHTML = `<tr><td colspan="${totalColumns}">No hay datos disponibles.</td></tr>`;
      return;
    }

    renderRows();
  };

  const renderActionableCenter = (report) => {
    const priority = document.getElementById('actionable-priority');
    const priorityCaption = document.getElementById('actionable-priority-caption');
    const quality = document.getElementById('actionable-quality');
    const qualityCaption = document.getElementById('actionable-quality-caption');
    const nextStep = document.getElementById('actionable-next-step');
    const nextStepCaption = document.getElementById('actionable-next-step-caption');

    if (!priority || !priorityCaption || !quality || !qualityCaption || !nextStep || !nextStepCaption) {
      return;
    }

    if (!report || !report.pipeline_health) {
      priority.textContent = 'Sin datos';
      priorityCaption.textContent = 'No hay reporte de ejecución disponible.';
      quality.textContent = 'Sin datos';
      qualityCaption.textContent = 'No se puede estimar calidad reciente.';
      nextStep.textContent = 'Revisar fuente de datos';
      nextStepCaption.textContent = 'Genera una nueva corrida para habilitar recomendaciones.';
      return;
    }

    const actions = report.summary?.actions || {};
    const buy = Number(actions.BUY || 0);
    const sell = Number(actions.SELL || 0);
    const hold = Number(actions.HOLD || 0);

    if (buy > sell && buy >= hold) {
      priority.textContent = 'Sesgo comprador';
      priorityCaption.textContent = `${buy} BUY vs ${sell} SELL en el corte actual.`;
    } else if (sell > buy && sell >= hold) {
      priority.textContent = 'Sesgo vendedor';
      priorityCaption.textContent = `${sell} SELL vs ${buy} BUY en el corte actual.`;
    } else {
      priority.textContent = 'Modo defensivo';
      priorityCaption.textContent = `${hold} HOLD; conviene priorizar control de riesgo.`;
    }

    const quality5d = report.summary?.quality_5d || {};
    const hit = Number(quality5d.Acierto || 0);
    const fail = Number(quality5d.Fallo || 0);
    const resolved = hit + fail;
    const hitRate = resolved > 0 ? (hit / resolved) * 100 : 0;
    quality.textContent = `${hitRate.toFixed(1)}% de acierto`;
    qualityCaption.textContent = resolved > 0
      ? `${hit} aciertos y ${fail} fallos (sin contar pendientes).`
      : 'Aún no hay resultados cerrados en ventana de 5 días.';

    const topRecommendation = (report.top_recommendations || [])[0] || null;
    if (topRecommendation) {
      nextStep.textContent = `${topRecommendation.action || 'HOLD'} ${topRecommendation.ticker || ''}`.trim();
      nextStepCaption.textContent = `Score ${Number(topRecommendation.strategy_score || 0).toFixed(4)} · resultado 5d: ${topRecommendation.result_5d || 'Pendiente'}.`;
    } else {
      nextStep.textContent = 'Sin recomendación prioritaria';
      nextStepCaption.textContent = 'No hay tickers destacados en la corrida más reciente.';
    }
  };

  const renderLastRunReport = (report) => {
    const runDate = document.getElementById('report-run-date');
    const status = document.getElementById('report-status');
    const success = document.getElementById('report-success');
    const fallback = document.getElementById('report-fallback');
    const artifactsList = document.getElementById('report-artifacts-list');
    const actionSummaryBody = document.getElementById('report-action-summary-body');
    const topRecBody = document.querySelector('#report-top-recommendations tbody');
    const modelMetricsBody = document.querySelector('#report-model-metrics tbody');
    const modelCoverageBody = document.querySelector('#report-model-coverage tbody');

    if (!runDate || !status || !success || !fallback || !artifactsList || !actionSummaryBody || !topRecBody || !modelMetricsBody || !modelCoverageBody) {
      return;
    }

    if (!report || !report.pipeline_health) {
      runDate.textContent = 'Sin datos';
      status.textContent = 'Sin datos';
      success.textContent = 'N/D';
      fallback.textContent = 'Sin datos';
      artifactsList.innerHTML = '<li>No hay artefactos de reporte disponibles.</li>';
      actionSummaryBody.innerHTML = '<tr><td>Sin datos disponibles.</td></tr>';
      topRecBody.innerHTML = '<tr><td colspan="7">Sin datos disponibles.</td></tr>';
      modelMetricsBody.innerHTML = '<tr><td colspan="5">Sin datos disponibles.</td></tr>';
      modelCoverageBody.innerHTML = '<tr><td colspan="3">Sin datos disponibles.</td></tr>';
      renderActionableCenter(null);
      return;
    }

    renderActionableCenter(report);

    const health = report.pipeline_health || {};
    const dataQuality = report.data_quality || {};
    const predictionsValid = dataQuality.predictions_valid !== false;
    const qualityCause = dataQuality.cause || 'ok';
    runDate.textContent = report.run_date || health.run_date || 'N/D';
    status.textContent = predictionsValid
      ? (health.status || 'N/D')
      : `${health.status || 'DEGRADADO'} · ⚠ Predicciones inválidas`;
    success.textContent = `${Number(health.success_pct || 0).toFixed(2)}% (${health.successful_steps || 0}/${health.total_steps || 0})`;
    fallback.textContent = health.fallback_offline || 'No detectado';

    const edgeCoverage = report.summary?.edge_coverage || {};
    const qualityWarning = predictionsValid
      ? ''
      : `<li style="color:#ffb020;font-weight:700;"><strong>⚠ Calidad de predicciones:</strong> inválida (${qualityCause}). Revisar artefacto del último run.</li>`;
    artifactsList.innerHTML = `
      ${qualityWarning}
      <li><strong>Predicciones:</strong> ${report.artifacts?.predictions_file || 'n/a'}</li>
      <li><strong>Predicciones válidas:</strong> ${predictionsValid ? 'Sí' : 'No'}</li>
      <li><strong>Métricas:</strong> ${report.artifacts?.metrics_file || 'n/a'}</li>
      <li><strong>Edge metrics:</strong> ${report.artifacts?.edge_metrics_file || 'n/a'}</li>
      <li><strong>Cobertura edge:</strong> ${edgeCoverage.rows || 0} filas, ${edgeCoverage.tickers || 0} tickers, ${edgeCoverage.models || 0} modelos.</li>
      <li><strong>Generado:</strong> ${report.generated_at || 'n/a'}</li>
    `;

    const actions = report.summary?.actions || {};
    const quality = report.summary?.quality_5d || {};
    actionSummaryBody.innerHTML = `
      <tr><td>BUY</td><td>${actions.BUY || 0}</td></tr>
      <tr><td>SELL</td><td>${actions.SELL || 0}</td></tr>
      <tr><td>HOLD</td><td>${actions.HOLD || 0}</td></tr>
      <tr><td>Aciertos 5d</td><td>${quality.Acierto || 0}</td></tr>
      <tr><td>Fallos 5d</td><td>${quality.Fallo || 0}</td></tr>
      <tr><td>Pendientes 5d</td><td>${quality.Pendiente || 0}</td></tr>
    `;

    const perModelCoverage = edgeCoverage.by_model || [];
    modelCoverageBody.innerHTML = perModelCoverage.length
      ? perModelCoverage.map((row) => `
          <tr>
            <td>${row.model || '-'}</td>
            <td>${row.tickers ?? 0}</td>
            <td>${row.rows ?? 0}</td>
          </tr>
        `).join('')
      : '<tr><td colspan="3">Sin datos disponibles.</td></tr>';

    const recommendations = report.top_recommendations || [];
    topRecBody.innerHTML = recommendations.length
      ? recommendations.map((row) => `
          <tr>
            <td>${row.ticker || '-'}</td>
            <td>${row.action || '-'}</td>
            <td>${Number(row.strategy_score || 0).toFixed(4)}</td>
            <td>${row.ret_1d || 'N/D'}</td>
            <td>${row.ret_5d || 'N/D'}</td>
            <td>${row.ret_20d || 'N/D'}</td>
            <td>${row.result_5d || 'Pendiente'}</td>
          </tr>
        `).join('')
      : '<tr><td colspan="7">Sin datos disponibles.</td></tr>';

    const metrics = report.model_metrics || [];
    modelMetricsBody.innerHTML = metrics.length
      ? metrics.map((row) => `
          <tr>
            <td>${row.model || '-'}</td>
            <td>${row.MAE !== undefined ? Number(row.MAE).toFixed(4) : '-'}</td>
            <td>${row.RMSE !== undefined ? Number(row.RMSE).toFixed(4) : '-'}</td>
            <td>${row.MAPE !== undefined ? Number(row.MAPE).toFixed(2) : '-'}</td>
            <td>${row.R2 !== undefined ? Number(row.R2).toFixed(4) : '-'}</td>
          </tr>
        `).join('')
      : '<tr><td colspan="5">Sin datos disponibles.</td></tr>';
  };


  const loadDashboardData = (versionToken) => Promise.all([
    fetch(`viz/pipeline_health.csv?v=${versionToken}`, { cache: 'no-store' })
      .then((r) => r.ok ? r.text() : '')
      .then((text) => renderPipelineHealth(parseCsv(text)))
      .catch(() => renderPipelineHealth([])),
    fetch(`viz/strategy_performance.csv?v=${versionToken}`, { cache: 'no-store' })
      .then((r) => r.ok ? r.text() : '')
      .then((text) => renderStrategyPerformance(parseCsv(text)))
      .catch(() => renderStrategyPerformance([])),
    fetch(`viz/action_recommendations.csv?v=${versionToken}`, { cache: 'no-store' })
      .then((r) => r.ok ? r.text() : '')
      .then((text) => renderActionRecommendations(parseCsv(text)))
      .catch(() => renderActionRecommendations([])),
    fetch(`viz/last_run_report.json?v=${versionToken}`, { cache: 'no-store' })
      .then((r) => r.ok ? r.json() : null)
      .then((report) => renderLastRunReport(report))
      .catch(() => renderLastRunReport(null)),
    fetch(`viz/strategy_performance/budget_and_action.csv?v=${versionToken}`, { cache: 'no-store' })
      .then((r) => r.ok ? r.text() : '')
      .then((text) => renderStrategyBudgetAction(parseCsv(text)))
      .catch(() => renderStrategyBudgetAction([])),
    fetch(`viz/strategy_performance/action_history.csv?v=${versionToken}`, { cache: 'no-store' })
      .then((r) => r.ok ? r.text() : '')
      .then((text) => renderStrategyActionHistory(parseCsv(text)))
      .catch(() => renderStrategyActionHistory([])),
  ]);

  fetch('viz/manifest.json', { cache: 'no-store' })
    .then((response) => response.ok ? response.json() : null)
    .then((manifest) => {
      if (!manifest || !manifest.generated_at) {
        return loadDashboardData(Date.now());
      }

      const windowDays = document.getElementById('window-days');
      const windowRange = document.getElementById('window-range');
      if (windowDays && manifest.window_days) {
        windowDays.textContent = String(manifest.window_days);
      }
      if (windowRange && manifest.window_start && manifest.window_end) {
        windowRange.textContent = `${manifest.window_start} a ${manifest.window_end}`;
      }

      images.forEach((image) => {
        const src = image.getAttribute('src');
        if (!src || src.includes('?')) {
          return;
        }
        image.setAttribute('src', `${src}?v=${manifest.generated_at}`);
      });

      return loadDashboardData(manifest.generated_at);
    })
    .catch(() => {
      renderPipelineHealth([]);
      renderStrategyPerformance([]);
      renderActionRecommendations([]);
      renderLastRunReport(null);
      renderStrategyBudgetAction([]);
      renderStrategyActionHistory([]);
      // Keep page functional even if manifest isn't available.
    });
});
