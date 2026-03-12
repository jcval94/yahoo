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
      sortKey: 'date',
      sortDirection: 'desc',
      filters: {},
      page: 1,
      pageSize: 20,
    };

    const parseComparable = (value, type) => {
      if (value === null || value === undefined || value === '') return null;
      if (type === 'number') {
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : null;
      }
      return String(value).toLowerCase();
    };

    const renderRows = () => {
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

      const filtered = rows.filter((row) => allColumns.every((column) => {
        const filterValue = (state.filters[column.key] || '').toString().trim().toLowerCase();
        if (!filterValue) return true;
        return String(getColumnValue(row, column.key) || '').toLowerCase().includes(filterValue);
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

    if (controls) {
      controls.innerHTML = `
        <label>
          Ordenar por
          <select id="action-sort-key">
            <option value="">Sin orden</option>
            ${allColumns.map((column) => `<option value="${column.key}">${column.label}</option>`).join('')}
          </select>
        </label>
        <label>
          Dirección
          <select id="action-sort-direction">
            <option value="asc">Ascendente</option>
            <option value="desc">Descendente</option>
          </select>
        </label>
        ${allColumns.map((column) => `
          <label>
            Filtrar ${column.label}
            <input type="text" id="filter-${column.key}" placeholder="Contiene..." />
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

      const sortKeySelect = controls.querySelector('#action-sort-key');
      const sortDirectionSelect = controls.querySelector('#action-sort-direction');
      const pageSizeSelect = controls.querySelector('#action-page-size');
      const showAllButton = controls.querySelector('#action-show-all');
      const prevButton = controls.querySelector('#action-page-prev');
      const nextButton = controls.querySelector('#action-page-next');

      if (pageSizeSelect) {
        pageSizeSelect.value = String(state.pageSize);
      }
      if (sortKeySelect) sortKeySelect.value = state.sortKey;
      if (sortDirectionSelect) sortDirectionSelect.value = state.sortDirection;

      sortKeySelect?.addEventListener('change', (event) => {
        state.sortKey = event.target.value;
        state.page = 1;
        renderRows();
      });

      sortDirectionSelect?.addEventListener('change', (event) => {
        state.sortDirection = event.target.value;
        state.page = 1;
        renderRows();
      });

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

      allColumns.forEach((column) => {
        const input = controls.querySelector(`#filter-${column.key}`);
        input?.addEventListener('input', (event) => {
          state.filters[column.key] = event.target.value;
          state.page = 1;
          renderRows();
        });
      });
    }

    tableHeadRow.innerHTML = [
      ...staticColumns.map((column) => `<th>${column.label}</th>`),
      ...modelGroups.map((group) => `<th class="model-prediction-col">${group.displayLabel.toUpperCase()}</th>`),
    ].join('');

    if (!rows.length) {
      const totalColumns = staticColumns.length + (modelGroups.length || 1);
      tableBody.innerHTML = `<tr><td colspan="${totalColumns}">No hay datos disponibles.</td></tr>`;
      return;
    }

    renderRows();
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
      return;
    }

    const health = report.pipeline_health || {};
    runDate.textContent = report.run_date || health.run_date || 'N/D';
    status.textContent = health.status || 'N/D';
    success.textContent = `${Number(health.success_pct || 0).toFixed(2)}% (${health.successful_steps || 0}/${health.total_steps || 0})`;
    fallback.textContent = health.fallback_offline || 'No detectado';

    const edgeCoverage = report.summary?.edge_coverage || {};
    artifactsList.innerHTML = `
      <li><strong>Predicciones:</strong> ${report.artifacts?.predictions_file || 'n/a'}</li>
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


  const buildInteractiveAnalytics = () => {
    const windowSelect = document.getElementById('analytics-window');
    const timeseriesSvg = document.getElementById('timeseries-chart');
    const timeseriesMeta = document.getElementById('timeseries-meta');
    const toggleContainer = document.getElementById('timeseries-toggles');
    const funnelSvg = document.getElementById('funnel-chart');
    const funnelMeta = document.getElementById('funnel-meta');
    const cohortGrid = document.getElementById('cohort-chart');
    const hourlyGrid = document.getElementById('hourly-heatmap');
    const geoSvg = document.getElementById('geo-chart');
    const geoMeta = document.getElementById('geo-meta');
    const scatterSvg = document.getElementById('scatter-chart');
    if (!windowSelect || !timeseriesSvg || !funnelSvg || !cohortGrid || !hourlyGrid || !geoSvg || !scatterSvg) return;

    const colors = { volume: '#2563eb', conversion: '#10b981', revenue: '#f59e0b' };
    const activeSeries = new Set(['volume', 'conversion']);
    const seriesLabels = { volume: 'Volumen', conversion: 'Conversión', revenue: 'Revenue' };

    const randomSeries = (days, base, wave, noise, trend = 0) => Array.from({ length: days }, (_, i) => (
      base + Math.sin(i / wave) * base * 0.15 + (Math.random() - 0.5) * noise + i * trend
    ));

    const renderTimeseries = (days) => {
      const data = {
        volume: randomSeries(days, 220, 4.2, 35, 0.12),
        conversion: randomSeries(days, 130, 5.5, 26, 0.05),
        revenue: randomSeries(days, 170, 6.2, 30, 0.09),
      };
      const maxVal = Math.max(...Object.values(data).flat());
      const w = 860;
      const h = 280;
      const pad = 28;
      const toPoint = (value, i) => {
        const x = pad + (i * (w - pad * 2)) / Math.max(1, days - 1);
        const y = h - pad - (value / maxVal) * (h - pad * 2);
        return `${x.toFixed(2)},${y.toFixed(2)}`;
      };
      const lines = [...activeSeries].map((key) => `
        <polyline fill="none" stroke="${colors[key]}" stroke-width="3" points="${data[key].map(toPoint).join(' ')}" />
      `).join('');
      timeseriesSvg.innerHTML = `<rect x="0" y="0" width="${w}" height="${h}" fill="transparent"/>${lines}`;
      const latest = [...activeSeries].map((k) => `${seriesLabels[k]}: ${data[k][days - 1].toFixed(1)}`).join(' · ');
      timeseriesMeta.textContent = `Ventana ${days}d · ${latest} · Colores: azul #2563EB, verde #10B981, ámbar #F59E0B`;
    };

    toggleContainer.innerHTML = Object.keys(seriesLabels).map((key) => `
      <button type="button" data-series-toggle="${key}" class="${activeSeries.has(key) ? 'is-active' : ''}">${seriesLabels[key]}</button>
    `).join('');
    toggleContainer.querySelectorAll('button').forEach((button) => {
      button.addEventListener('click', () => {
        const key = button.dataset.seriesToggle;
        if (activeSeries.has(key)) activeSeries.delete(key); else activeSeries.add(key);
        if (!activeSeries.size) activeSeries.add('volume');
        toggleContainer.querySelectorAll('button').forEach((b) => b.classList.toggle('is-active', activeSeries.has(b.dataset.seriesToggle)));
        renderTimeseries(Number(windowSelect.value));
      });
    });

    const funnelStages = [
      { label: 'Visitas', value: 10000 },
      { label: 'Registro', value: 6200 },
      { label: 'Activación', value: 4100 },
      { label: 'Compra', value: 2300 },
    ];

    const renderFunnel = () => {
      const max = funnelStages[0].value;
      funnelSvg.innerHTML = funnelStages.map((stage, idx) => {
        const y = 20 + idx * 56;
        const width = (stage.value / max) * 320;
        return `<g data-stage-index="${idx}" style="cursor:pointer"><rect x="20" y="${y}" width="${width}" height="36" rx="8" fill="${idx === 3 ? '#ef4444' : '#3b82f6'}" opacity="${0.9 - idx * 0.14}"/><text x="30" y="${y + 23}" fill="#e2e8f0" font-size="13">${stage.label}: ${stage.value}</text></g>`;
      }).join('');
      funnelMeta.textContent = 'Click en una etapa para ver pérdida relativa frente a la etapa anterior.';
      funnelSvg.querySelectorAll('g').forEach((g) => g.addEventListener('click', () => {
        const idx = Number(g.getAttribute('data-stage-index'));
        if (idx === 0) {
          funnelMeta.textContent = `Etapa ${funnelStages[idx].label}: base de referencia.`;
          return;
        }
        const drop = (1 - funnelStages[idx].value / funnelStages[idx - 1].value) * 100;
        funnelMeta.textContent = `${funnelStages[idx].label}: caída de ${drop.toFixed(1)}% vs ${funnelStages[idx - 1].label}.`;
      }));
    };

    const colorScale = (ratio) => {
      const v = Math.max(0, Math.min(1, ratio));
      const b = Math.round(245 - v * 120);
      return `rgb(${30 + Math.round(v * 20)}, ${80 + Math.round(v * 70)}, ${b})`;
    };

    const renderCohort = () => {
      cohortGrid.innerHTML = '';
      for (let row = 0; row < 12; row += 1) {
        for (let col = 0; col < 12; col += 1) {
          const decay = Math.max(0, 0.88 - col * 0.06 - row * 0.015 + Math.random() * 0.03);
          const cell = document.createElement('div');
          cell.className = 'heatmap-cell';
          cell.style.background = colorScale(decay);
          cell.title = `Cohorte ${row + 1}, mes ${col + 1}: ${(decay * 100).toFixed(1)}%`;
          cohortGrid.appendChild(cell);
        }
      }
    };

    const renderHourly = () => {
      hourlyGrid.innerHTML = '';
      for (let day = 0; day < 7; day += 1) {
        for (let hour = 0; hour < 24; hour += 1) {
          const peak = Math.exp(-((hour - 15) ** 2) / 40) * 0.65 + Math.exp(-((hour - 10) ** 2) / 20) * 0.35;
          const score = Math.max(0, Math.min(1, peak + Math.random() * 0.2 - 0.05 + day * 0.02));
          const cell = document.createElement('div');
          cell.className = 'heatmap-cell';
          cell.style.background = colorScale(score);
          cell.title = `Día ${day + 1}, ${hour}:00 = ${(score * 100).toFixed(1)}`;
          hourlyGrid.appendChild(cell);
        }
      }
    };

    const geoPoints = [
      { name: 'Norteamérica', x: 80, y: 90, value: 82 },
      { name: 'LatAm', x: 130, y: 170, value: 61 },
      { name: 'Europa', x: 220, y: 82, value: 75 },
      { name: 'África', x: 230, y: 145, value: 58 },
      { name: 'Asia', x: 305, y: 110, value: 92 },
      { name: 'Oceanía', x: 355, y: 188, value: 46 },
    ];

    const renderGeo = () => {
      geoSvg.innerHTML = `<rect x="12" y="16" width="396" height="228" rx="14" fill="rgba(15,23,42,0.55)" stroke="rgba(148,163,184,0.35)"/>` + geoPoints.map((point) => (
        `<g data-region="${point.name}" style="cursor:pointer"><circle cx="${point.x}" cy="${point.y}" r="${8 + point.value / 12}" fill="#10b981" opacity="0.8"/><text x="${point.x + 10}" y="${point.y + 4}" fill="#e2e8f0" font-size="11">${point.name}</text></g>`
      )).join('');
      geoMeta.textContent = 'Click en una región para ver su contribución.';
      geoSvg.querySelectorAll('g').forEach((node) => node.addEventListener('click', () => {
        const region = geoPoints.find((p) => p.name === node.getAttribute('data-region'));
        geoMeta.textContent = `${region.name}: índice de contribución ${region.value}/100 en la ventana seleccionada.`;
      }));
    };

    const channels = [
      { label: 'Orgánico', color: '#2563eb' },
      { label: 'Paid', color: '#10b981' },
      { label: 'Referral', color: '#f59e0b' },
      { label: 'Email', color: '#8b5cf6' },
      { label: 'Social', color: '#ef4444' },
    ];

    const renderScatter = () => {
      const dots = Array.from({ length: 45 }, (_, idx) => {
        const channel = channels[idx % channels.length];
        const x = 50 + Math.random() * 760;
        const y = 30 + Math.random() * 220;
        const r = 4 + Math.random() * 10;
        return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="${channel.color}" fill-opacity="0.68"><title>${channel.label}</title></circle>`;
      }).join('');
      scatterSvg.innerHTML = `<line x1="44" y1="250" x2="820" y2="250" stroke="#94a3b8"/><line x1="44" y1="26" x2="44" y2="250" stroke="#94a3b8"/>${dots}`;
    };

    const rerenderAll = () => {
      renderTimeseries(Number(windowSelect.value));
      renderFunnel();
      renderCohort();
      renderHourly();
      renderGeo();
      renderScatter();
    };

    windowSelect.addEventListener('change', rerenderAll);
    rerenderAll();
  };

  buildInteractiveAnalytics();

  fetch('viz/manifest.json', { cache: 'no-store' })
    .then((response) => response.ok ? response.json() : null)
    .then((manifest) => {
      if (!manifest || !manifest.generated_at) {
        return;
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

      return Promise.all([
        fetch(`viz/pipeline_health.csv?v=${manifest.generated_at}`, { cache: 'no-store' })
          .then((r) => r.ok ? r.text() : '')
          .then((text) => renderPipelineHealth(parseCsv(text)))
          .catch(() => renderPipelineHealth([])),
        fetch(`viz/strategy_performance.csv?v=${manifest.generated_at}`, { cache: 'no-store' })
          .then((r) => r.ok ? r.text() : '')
          .then((text) => renderStrategyPerformance(parseCsv(text)))
          .catch(() => renderStrategyPerformance([])),
        fetch(`viz/action_recommendations.csv?v=${manifest.generated_at}`, { cache: 'no-store' })
          .then((r) => r.ok ? r.text() : '')
          .then((text) => renderActionRecommendations(parseCsv(text)))
          .catch(() => renderActionRecommendations([])),
        fetch(`viz/last_run_report.json?v=${manifest.generated_at}`, { cache: 'no-store' })
          .then((r) => r.ok ? r.json() : null)
          .then((report) => renderLastRunReport(report))
          .catch(() => renderLastRunReport(null)),
        fetch(`viz/strategy_performance/budget_and_action.csv?v=${manifest.generated_at}`, { cache: 'no-store' })
          .then((r) => r.ok ? r.text() : '')
          .then((text) => renderStrategyBudgetAction(parseCsv(text)))
          .catch(() => renderStrategyBudgetAction([])),
        fetch(`viz/strategy_performance/action_history.csv?v=${manifest.generated_at}`, { cache: 'no-store' })
          .then((r) => r.ok ? r.text() : '')
          .then((text) => renderStrategyActionHistory(parseCsv(text)))
          .catch(() => renderStrategyActionHistory([])),
      ]);
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
