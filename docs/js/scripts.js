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
          </tr>
        `)
        .join('')
      : '<tr><td colspan="5">No hay datos disponibles.</td></tr>';

    strategyTables.forEach((tableBody) => {
      tableBody.innerHTML = tableMarkup;
    });
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

    const canonicalModelName = (columnName) => {
      const rawModel = columnName.replace('pred_', '');
      const normalized = rawModel
        .toLowerCase()
        .replace(/model|regressor/g, '')
        .replace(/[^a-z0-9]/g, '');
      return normalized || rawModel.toLowerCase();
    };

    const scoreModelColumn = (columnName) => {
      const rawModel = columnName.replace('pred_', '');
      let score = 0;
      if (/^[a-z0-9_]+$/.test(rawModel)) score += 2;
      if (!/(model|regressor)/i.test(rawModel)) score += 1;
      score -= rawModel.length / 100;
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
            sourceColumns: [columnName],
          });
          return groups;
        }

        existing.sourceColumns.push(columnName);
        if (scoreModelColumn(columnName) > scoreModelColumn(existing.labelColumn)) {
          existing.labelColumn = columnName;
        }
        return groups;
      }, []);

    const allColumns = [
      ...staticColumns,
      ...modelGroups.map((group) => ({
        key: group.canonicalName,
        label: group.labelColumn.replace('pred_', '').toUpperCase(),
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
          .sort((a, b) => scoreModelColumn(b) - scoreModelColumn(a));
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
      ...modelGroups.map((group) => `<th class="model-prediction-col">${group.labelColumn.replace('pred_', '').toUpperCase()}</th>`),
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
      ]);
    })
    .catch(() => {
      renderPipelineHealth([]);
      renderStrategyPerformance([]);
      renderActionRecommendations([]);
      renderLastRunReport(null);
      // Keep page functional even if manifest isn't available.
    });
});
