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

  const tabButtons = document.querySelectorAll('.tab-button');
  const tabPanels = document.querySelectorAll('.tab-panel');
  tabButtons.forEach((button) => {
    button.addEventListener('click', () => {
      const target = button.getAttribute('data-tab-target');
      tabButtons.forEach((b) => b.classList.toggle('is-active', b === button));
      tabPanels.forEach((panel) => {
        const panelName = panel.getAttribute('data-tab-panel');
        panel.classList.toggle('is-hidden', panelName !== target);
      });
    });
  });

  const images = document.querySelectorAll('img');
  images.forEach((image) => {
    image.addEventListener('error', () => {
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
    const table = document.querySelector('#strategy-performance-table tbody');
    if (!table) return;
    if (!rows.length) {
      table.innerHTML = '<tr><td colspan="5">No hay datos disponibles.</td></tr>';
      return;
    }

    table.innerHTML = rows
      .map((row) => `
        <tr>
          <td>${row.strategy}</td>
          <td>$${Number(row.ending_equity || 0).toFixed(2)}</td>
          <td>${Number(row.return_pct || 0).toFixed(2)}%</td>
          <td>${Number(row.win_rate || 0).toFixed(2)}%</td>
          <td>${Number(row.max_drawdown || 0).toFixed(4)}</td>
        </tr>
      `)
      .join('');
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
      { key: 'date', label: 'Fecha' },
      { key: 'ticker', label: 'Ticker' },
      { key: 'best_model', label: 'Modelo líder' },
      { key: 'strategy_score', label: 'Score', formatter: (value) => Number(value || 0).toFixed(4) },
      { key: 'action', label: 'Acción' },
      { key: 'ret_1d', label: 'Resultado 1d' },
      { key: 'ret_5d', label: 'Resultado 5d' },
      { key: 'ret_20d', label: 'Resultado 20d' },
      { key: 'result_5d', label: 'Calidad 5d' },
    ];

    const modelColumns = Array.from(
      rows.reduce((allColumns, row) => {
        Object.keys(row || {}).forEach((key) => {
          if (key.startsWith('pred_')) {
            allColumns.add(key);
          }
        });
        return allColumns;
      }, new Set())
    ).sort((a, b) => a.localeCompare(b));

    tableHeadRow.innerHTML = [
      ...staticColumns.map((column) => `<th>${column.label}</th>`),
      ...modelColumns.map((col) => `<th class="model-prediction-col">${col.replace('pred_', '').toUpperCase()}</th>`),
    ].join('');

    if (!rows.length) {
      const totalColumns = staticColumns.length + (modelColumns.length || 1);
      tableBody.innerHTML = `<tr><td colspan="${totalColumns}">No hay datos disponibles.</td></tr>`;
      return;
    }

    tableBody.innerHTML = rows
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

        const modelCells = modelColumns.map((col) => `<td class="model-predictions">${row[col] || '-'}</td>`);

        return `
          <tr>
            ${staticCells.join('')}
            ${modelCells.join('')}
          </tr>
        `;
      })
      .join('');
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

    if (!runDate || !status || !success || !fallback || !artifactsList || !actionSummaryBody || !topRecBody || !modelMetricsBody) {
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
