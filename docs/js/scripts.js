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

  const renderActionRecommendations = (rows) => {
    const table = document.querySelector('#action-recommendations-table tbody');
    if (!table) return;
    if (!rows.length) {
      table.innerHTML = '<tr><td colspan="6">No hay datos disponibles.</td></tr>';
      return;
    }

    const modelColumns = Object.keys(rows[0] || {})
      .filter((key) => key.startsWith('pred_'))
      .sort((a, b) => a.localeCompare(b));

    table.innerHTML = rows
      .map((row) => {
        const action = (row.action || 'HOLD').toUpperCase();
        const cls = action === 'BUY' ? 'action-buy' : action === 'SELL' ? 'action-sell' : 'action-hold';
        const modelSignals = modelColumns
          .map((col) => `${col.replace('pred_', '').toUpperCase()}: ${row[col] || '-'}`)
          .join(' · ');
        return `
          <tr>
            <td>${row.date || '-'}</td>
            <td>${row.ticker || '-'}</td>
            <td>${row.best_model || '-'}</td>
            <td>${Number(row.strategy_score || 0).toFixed(4)}</td>
            <td><span class="action-badge ${cls}">${action}</span></td>
            <td class="model-predictions">${modelSignals || '-'}</td>
          </tr>
        `;
      })
      .join('');
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
        fetch(`viz/strategy_performance.csv?v=${manifest.generated_at}`, { cache: 'no-store' })
          .then((r) => r.ok ? r.text() : '')
          .then((text) => renderStrategyPerformance(parseCsv(text)))
          .catch(() => renderStrategyPerformance([])),
        fetch(`viz/action_recommendations.csv?v=${manifest.generated_at}`, { cache: 'no-store' })
          .then((r) => r.ok ? r.text() : '')
          .then((text) => renderActionRecommendations(parseCsv(text)))
          .catch(() => renderActionRecommendations([])),
      ]);
    })
    .catch(() => {
      renderStrategyPerformance([]);
      renderActionRecommendations([]);
      // Keep page functional even if manifest isn't available.
    });
});
