/* 页内交互函数图渲染器：```plot 围栏块（data-plot JSON）→ Canvas 交互图。
   交互模式参照 sigmoid 动画：悬停十字线 + 实时读数 + 渐近线 + 关键点；
   额外支持多曲线、入场描线动画、亮/暗主题自适应、触屏。 */
(function () {
  'use strict';

  function dark() { return document.documentElement.getAttribute('data-theme') === 'dark'; }
  function cv(name, fb) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fb;
  }
  function fmt(v) { return Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3); }

  function initPlot(el) {
    if (el.__plotInit) return;
    el.__plotInit = true;
    var cfg;
    try { cfg = JSON.parse(el.getAttribute('data-plot')); }
    catch (e) { el.textContent = '[plot 配置异常]'; return; }

    // --- DOM：标题头 + 画布 + 图例 ---
    el.innerHTML = '';
    var head = document.createElement('div'); head.className = 'ip-head';
    var t = document.createElement('span'); t.className = 'ip-title'; t.textContent = cfg.title || '';
    head.appendChild(t);
    if (cfg.subtitle) { var sub = document.createElement('span'); sub.className = 'ip-sub'; sub.textContent = cfg.subtitle; head.appendChild(sub); }
    var wrap = document.createElement('div'); wrap.className = 'ip-wrap';
    var cvs = document.createElement('canvas'); wrap.appendChild(cvs);
    var legend = document.createElement('div'); legend.className = 'ip-legend';
    el.appendChild(head); el.appendChild(wrap); el.appendChild(legend);
    if (cfg.note) { var note = document.createElement('div'); note.className = 'ip-note'; note.textContent = cfg.note; el.appendChild(note); }

    var curves = (cfg.curves || []).map(function (c, i) {
      var item = document.createElement('span'); item.className = 'ip-item';
      var dot = document.createElement('i'); dot.className = 'ip-dot';
      var lab = document.createElement('span'); lab.textContent = c.label || ('f' + (i + 1));
      var val = document.createElement('code'); val.textContent = '—';
      item.appendChild(dot); item.appendChild(lab); item.appendChild(val);
      legend.appendChild(item);
      return { fn: new Function('x', 'return (' + (c.expr || '0') + ');'),
               label: c.label || ('f' + (i + 1)), valEl: val, dotEl: dot, col: null };
    });

    var ctx = cvs.getContext('2d');
    var reveal = 0, hover = null, raf = null, layout = null;

    function palette(i) {
      var p = [cv('--accent', '#0969da'), cv('--success', '#1a7f37'),
               dark() ? '#d2a8ff' : '#8250df', cv('--warn', '#9a6700')];
      return p[i % p.length];
    }

    function ranges() {
      var R = { x0: (cfg.x || [-6, 6])[0], x1: (cfg.x || [-6, 6])[1], y0: -1, y1: 1 };
      if (cfg.y) { R.y0 = cfg.y[0]; R.y1 = cfg.y[1]; }
      else {
        var mn = Infinity, mx = -Infinity;
        curves.forEach(function (c) {
          for (var k = 0; k <= 200; k++) {
            var v; try { v = c.fn(R.x0 + (R.x1 - R.x0) * k / 200); } catch (e) { continue; }
            if (isFinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v; }
          }
        });
        if (!isFinite(mn)) { mn = -1; mx = 1; }
        if (mx - mn < 1e-6) { mn -= 1; mx += 1; }
        var pad = (mx - mn) * 0.12; R.y0 = mn - pad; R.y1 = mx + pad;
      }
      return R;
    }

    function draw() {
      var rect = wrap.getBoundingClientRect();
      if (rect.width < 10) return;
      var w = rect.width, h = rect.height || w * 0.5625;
      var dpr = window.devicePixelRatio || 1;
      var pw = Math.round(w * dpr), ph = Math.round(h * dpr);
      if (cvs.width !== pw || cvs.height !== ph) { cvs.width = pw; cvs.height = ph; }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      var R = ranges();
      var pad = { top: 18, bottom: 26, left: 46, right: 16 };
      var plotW = w - pad.left - pad.right, plotH = h - pad.top - pad.bottom;
      if (plotW < 20 || plotH < 20) return;
      function X(x) { return pad.left + (x - R.x0) / (R.x1 - R.x0) * plotW; }
      function Y(y) { return pad.top + plotH - (y - R.y0) / (R.y1 - R.y0) * plotH; }
      layout = { R: R, pad: pad, plotW: plotW, plotH: plotH, w: w, h: h };

      var isDark = dark();
      var cFg = cv('--fg', '#1f2328'), cFg3 = cv('--fg3', '#59636e'), cFg4 = cv('--fg4', '#818b98');
      var cGrid = isDark ? 'rgba(240,246,252,0.08)' : '#eef1f4';
      var cDanger = isDark ? '#ff7b72' : '#cf222e';
      var mono = cv('--mono', 'monospace');
      curves.forEach(function (c, i) { c.col = palette(i); c.dotEl.style.background = c.col; });

      ctx.clearRect(0, 0, w, h);

      // --- 网格 + 刻度 ---
      var xs = cfg.xticks || Math.max(0.5, +(((R.x1 - R.x0) / 8).toPrecision(2)));
      var ys = cfg.yticks || Math.max(0.25, +(((R.y1 - R.y0) / 5).toPrecision(2)));
      function nice(v, step) { return Math.abs(v / step) < 1e-6 ? 0 : Math.abs(step) < 1 ? v.toFixed(2) : String(Math.round(v * 100) / 100); }
      ctx.strokeStyle = cGrid; ctx.lineWidth = 1;
      ctx.font = '11px ' + mono; ctx.fillStyle = cFg4;
      var axisY = Math.min(Math.max(Y(0), pad.top), pad.top + plotH);   // y=0 轴（越界则贴边）
      var axisX = Math.min(Math.max(X(0), pad.left), pad.left + plotW);
      for (var gx = Math.ceil(R.x0 / xs) * xs; gx <= R.x1 + 1e-9; gx += xs) {
        var px = X(gx);
        ctx.beginPath(); ctx.moveTo(px, pad.top); ctx.lineTo(px, pad.top + plotH); ctx.stroke();
        ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        ctx.fillText(nice(gx, xs), px, Math.min(axisY + 5, pad.top + plotH + 6));
      }
      for (var gy = Math.ceil(R.y0 / ys) * ys; gy <= R.y1 + 1e-9; gy += ys) {
        var py = Y(gy);
        ctx.beginPath(); ctx.moveTo(pad.left, py); ctx.lineTo(pad.left + plotW, py); ctx.stroke();
        ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        ctx.fillText(nice(gy, ys), pad.left - 7, py);
      }

      // --- 坐标轴 ---
      ctx.strokeStyle = cFg3; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(pad.left, axisY); ctx.lineTo(pad.left + plotW, axisY); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(axisX, pad.top); ctx.lineTo(axisX, pad.top + plotH); ctx.stroke();

      // --- 渐近线 / 参考线（虚线）---
      (cfg.hlines || []).forEach(function (hl) {
        var hy = Y(hl.y);
        if (hy < pad.top || hy > pad.top + plotH) return;
        ctx.setLineDash([4, 5]); ctx.strokeStyle = cDanger; ctx.lineWidth = 1.3;
        ctx.beginPath(); ctx.moveTo(pad.left, hy); ctx.lineTo(pad.left + plotW, hy); ctx.stroke();
        ctx.setLineDash([]);
        if (hl.label) {
          ctx.fillStyle = cDanger; ctx.font = '11px ' + mono;
          ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
          ctx.fillText(hl.label, pad.left + 6, hy - 3);
        }
      });

      // --- 曲线（reveal 入场描线：只画到 x 进度处）---
      var xEnd = R.x0 + (R.x1 - R.x0) * reveal;
      curves.forEach(function (c) {
        ctx.beginPath(); ctx.strokeStyle = c.col; ctx.lineWidth = 2.6;
        ctx.lineJoin = 'round'; ctx.lineCap = 'round';
        var started = false;
        for (var k = 0; k <= 300; k++) {
          var x = R.x0 + (xEnd - R.x0) * k / 300;
          var y; try { y = c.fn(x); } catch (e) { started = false; continue; }
          if (!isFinite(y)) { started = false; continue; }
          var yy = Math.min(Math.max(Y(y), pad.top - 20), pad.top + plotH + 20);
          if (!started) { ctx.moveTo(X(x), yy); started = true; } else { ctx.lineTo(X(x), yy); }
        }
        ctx.stroke();
      });

      // --- 关键点（随 reveal 淡入）---
      ctx.save(); ctx.globalAlpha = Math.max(0, reveal * 2 - 1);
      (cfg.points || []).forEach(function (p) {
        var px = X(p[0]), py = Y(p[1]);
        ctx.fillStyle = cFg;
        ctx.beginPath(); ctx.arc(px, py, 4.5, 0, 2 * Math.PI); ctx.fill();
        ctx.strokeStyle = cv('--card', '#fff'); ctx.lineWidth = 2; ctx.stroke();
        if (p[2]) {
          ctx.fillStyle = cFg3; ctx.font = '12px ' + mono;
          ctx.textAlign = 'left';
          if (p[3] === 'below') { ctx.textBaseline = 'top'; ctx.fillText(p[2], px + 9, py + 8); }
          else { ctx.textBaseline = 'bottom'; ctx.fillText(p[2], px + 9, py - 4); }
        }
      });
      ctx.restore();

      // --- 悬停：十字线 + 各曲线取值点 + 悬浮读数框 ---
      if (hover !== null && hover >= R.x0 && hover <= R.x1) {
        ctx.setLineDash([3, 5]); ctx.strokeStyle = curves.length ? curves[0].col : cFg3;
        ctx.globalAlpha = 0.35; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(X(hover), pad.top); ctx.lineTo(X(hover), pad.top + plotH); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad.left, 0); ctx.lineTo(pad.left, 0); ctx.stroke();
        ctx.setLineDash([]); ctx.globalAlpha = 1;
        var rows = ['x = ' + hover.toFixed(2)];
        var cols = [null];
        curves.forEach(function (c) {
          var v; try { v = c.fn(hover); } catch (e) { v = NaN; }
          rows.push((c.label.length > 18 ? c.label.slice(0, 17) + '…' : c.label) + ' = ' + (isFinite(v) ? fmt(v) : '—'));
          cols.push(c.col);
          c.valEl.textContent = isFinite(v) ? fmt(v) : '—';
          if (isFinite(v)) {
            var py2 = Math.min(Math.max(Y(v), pad.top), pad.top + plotH);
            ctx.fillStyle = c.col;
            ctx.beginPath(); ctx.arc(X(hover), py2, 5, 0, 2 * Math.PI); ctx.fill();
            ctx.strokeStyle = cv('--card', '#fff'); ctx.lineWidth = 2; ctx.stroke();
          }
        });
        // 读数框（顶部，按光标左右分侧，避免遮挡曲线）
        ctx.font = '12px ' + mono;
        var tw = 0; rows.forEach(function (r) { tw = Math.max(tw, ctx.measureText(r).width); });
        tw += 46;
        var th = rows.length * 17 + 12;
        var lx = X(hover) > pad.left + plotW / 2 ? pad.left + 6 : pad.left + plotW - tw - 6;
        var ly = pad.top + 6;
        ctx.fillStyle = isDark ? 'rgba(22,27,34,0.92)' : 'rgba(255,255,255,0.92)';
        ctx.strokeStyle = cv('--line', '#d1d9e0'); ctx.lineWidth = 1;
        if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(lx, ly, tw, th, 6); ctx.fill(); ctx.stroke(); }
        else { ctx.strokeRect(lx, ly, tw, th); ctx.fillRect(lx, ly, tw, th); }
        rows.forEach(function (r, ri) {
          var cy2 = ly + 14 + ri * 17;
          if (cols[ri]) { ctx.fillStyle = cols[ri]; ctx.fillRect(lx + 10, cy2 - 3.5, 7, 7); }
          ctx.fillStyle = ri === 0 ? cFg : cFg3;
          ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
          ctx.fillText(r, lx + (cols[ri] ? 24 : 10), cy2);
        });
      } else {
        curves.forEach(function (c) { c.valEl.textContent = '—'; });
      }
    }

    // --- 入场描线动画 ---
    function animate() {
      if (raf) return;
      var t0 = null;
      function step(ts) {
        if (t0 === null) t0 = ts;
        var p = Math.min(1, (ts - t0) / 700);
        reveal = 1 - Math.pow(1 - p, 3);
        draw();
        if (p < 1) raf = requestAnimationFrame(step); else raf = null;
      }
      raf = requestAnimationFrame(step);
    }

    // --- 事件：悬停 / 触屏 ---
    function onPoint(clientX) {
      if (!layout) return;
      var r = cvs.getBoundingClientRect();
      var px = clientX - r.left;
      var R = layout.R;
      if (px < layout.pad.left || px > layout.pad.left + layout.plotW) { hover = null; }
      else { hover = R.x0 + (px - layout.pad.left) / layout.plotW * (R.x1 - R.x0); }
      draw();
    }
    function off() { hover = null; draw(); }
    cvs.addEventListener('mousemove', function (e) { onPoint(e.clientX); });
    cvs.addEventListener('mouseleave', off);
    cvs.addEventListener('touchmove', function (e) { e.preventDefault(); onPoint(e.touches[0].clientX); }, { passive: false });
    cvs.addEventListener('touchend', off);

    // --- 尺寸 / 主题 / 字号 变化重绘 ---
    if ('ResizeObserver' in window) { new ResizeObserver(function () { draw(); }).observe(wrap); }
    else { window.addEventListener('resize', function () { draw(); }); }
    new MutationObserver(function () { draw(); })
      .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'data-fs'] });

    // --- 首次进入视口 → 描线动画 ---
    if ('IntersectionObserver' in window) {
      var io = new IntersectionObserver(function (en) {
        if (en[0].isIntersecting) { io.disconnect(); animate(); }
      }, { rootMargin: '80px' });
      io.observe(el);
    } else { reveal = 1; }

    draw();
  }

  /* ---------- ```chunk：训练样本可视化（x token 色块 → y 高亮）---------- */
  function initChunk(el) {
    if (el.__plotInit) return;
    el.__plotInit = true;
    var cfg;
    try { cfg = JSON.parse(el.getAttribute('data-chunk')); }
    catch (e) { el.textContent = '[chunk 配置异常]'; return; }
    var seq = cfg.sequence || [];
    var n = Math.min(cfg.samples || seq.length - 1, seq.length - 1);

    el.innerHTML = '';
    var head = document.createElement('div'); head.className = 'cv-head';
    var t = document.createElement('span'); t.className = 'cv-title'; t.textContent = cfg.title || '';
    head.appendChild(t);
    el.appendChild(head);
    if (cfg.subtitle) {
      var sub = document.createElement('div'); sub.className = 'cv-sub'; sub.textContent = cfg.subtitle;
      el.appendChild(sub);
    }
    var list = document.createElement('div'); list.className = 'cv-list';
    for (var i = 0; i < n; i++) {
      var row = document.createElement('div'); row.className = 'cv-row';
      row.style.transitionDelay = (i * 45) + 'ms';
      var id = document.createElement('span'); id.className = 'cv-id'; id.textContent = '#' + (i + 1);
      var xs = document.createElement('div'); xs.className = 'cv-xs';
      for (var j = 0; j <= i; j++) {
        var tok = document.createElement('span');
        tok.className = 'cv-tok' + (j === i ? ' is-new' : '');
        tok.textContent = seq[j];
        xs.appendChild(tok);
      }
      var arrow = document.createElement('span'); arrow.className = 'cv-arrow'; arrow.textContent = '→';
      var yl = document.createElement('span'); yl.className = 'cv-yl'; yl.textContent = 'y=';
      var yv = document.createElement('span'); yv.className = 'cv-y'; yv.textContent = seq[i + 1];
      row.appendChild(id); row.appendChild(xs); row.appendChild(arrow);
      row.appendChild(yl); row.appendChild(yv);
      list.appendChild(row);
    }
    el.appendChild(list);
    var lg = document.createElement('div'); lg.className = 'cv-legend';
    lg.innerHTML = '<span class="cv-lg"><i class="cv-sw sw-old"></i>已有 token</span>'
      + '<span class="cv-lg"><i class="cv-sw sw-new"></i>新增 token ✦</span>'
      + '<span class="cv-lg"><i class="cv-sw sw-y"></i>输出 y</span>';
    el.appendChild(lg);
    if ('IntersectionObserver' in window) {
      var io = new IntersectionObserver(function (en) {
        if (en[0].isIntersecting) { io.disconnect(); el.classList.add('cv-in'); }
      }, { rootMargin: '60px' });
      io.observe(el);
    } else { el.classList.add('cv-in'); }
  }

  /* ---------- ```viz type=tree：树状结构（JS 测量布局 + SVG 连接线）---------- */
  function initTree(el) {
    if (el.__plotInit) return;
    el.__plotInit = true;
    var cfg;
    try { cfg = JSON.parse(el.getAttribute('data-viz')); }
    catch (e) { el.textContent = '[viz 配置异常]'; return; }

    el.innerHTML = '';
    var KIND_CLS = { input: 'viz-k-input', op: 'viz-k-op', mid: 'viz-k-mid', output: 'viz-k-output' };
    var KIND_LABEL = { input: '输入', op: '变换', mid: '中间结果', output: '输出' };
    var legendOv = cfg.legend || {};

    if (cfg.title || cfg.subtitle) {
      var head = document.createElement('div'); head.className = 'viz-head';
      var t = document.createElement('span'); t.className = 'viz-title'; t.textContent = cfg.title || '';
      head.appendChild(t);
      if (cfg.subtitle) { var s = document.createElement('span'); s.className = 'viz-sub'; s.textContent = cfg.subtitle; head.appendChild(s); }
      el.appendChild(head);
    }
    var stage = document.createElement('div'); stage.className = 'tr-stage';
    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'tr-svg');
    stage.appendChild(svg);
    var levelEls = [];
    var used = {};
    (cfg.levels || []).forEach(function (level) {
      var row = document.createElement('div'); row.className = 'tr-level';
      level.forEach(function (item) {
        var text = item[0], kind = item[1] || 'mid';
        used[kind] = 1;
        var slot = document.createElement('div'); slot.className = 'tr-slot';
        var chip = document.createElement('span');
        chip.className = 'viz-node ' + (KIND_CLS[kind] || 'viz-k-mid');
        chip.textContent = text;
        slot.appendChild(chip);
        row.appendChild(slot);
      });
      stage.appendChild(row);
      levelEls.push(row);
    });
    el.appendChild(stage);

    var legend = document.createElement('div'); legend.className = 'viz-legend';
    Object.keys(used).forEach(function (k) {
      var item = document.createElement('span'); item.className = 'cv-lg';
      var sw = document.createElement('i'); sw.className = 'cv-sw ' + (KIND_CLS[k] || 'viz-k-mid');
      var lb = document.createElement('span'); lb.textContent = legendOv[k] || KIND_LABEL[k] || k;
      item.appendChild(sw); item.appendChild(lb);
      legend.appendChild(item);
    });
    el.appendChild(legend);
    if (cfg.src) {
      var det = document.createElement('details'); det.className = 'viz-src';
      var sum = document.createElement('summary'); sum.textContent = '原始文本';
      var pre = document.createElement('pre'); pre.textContent = cfg.src;
      det.appendChild(sum); det.appendChild(pre);
      el.appendChild(det);
    }

    function draw() {
      svg.innerHTML = '';
      var st = stage.getBoundingClientRect();
      if (st.width < 10) return;
      svg.setAttribute('viewBox', '0 0 ' + st.width + ' ' + st.height);
      var chipPos = levelEls.map(function (row) {
        return Array.from(row.querySelectorAll('.viz-node')).map(function (c) {
          var r = c.getBoundingClientRect();
          return { x: r.left - st.left + r.width / 2, top: r.top - st.top, bottom: r.top - st.top + r.height };
        });
      });
      var stroke = cv('--line2', '#afb8c1');
      for (var i = 0; i + 1 < chipPos.length; i++) {
        var parents = chipPos[i], kids = chipPos[i + 1];
        parents.forEach(function (p, pi) {
          [2 * pi, 2 * pi + 1].forEach(function (ki) {
            if (!kids[ki]) return;
            var line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', p.x); line.setAttribute('y1', p.bottom);
            line.setAttribute('x2', kids[ki].x); line.setAttribute('y2', kids[ki].top);
            line.setAttribute('stroke', stroke); line.setAttribute('stroke-width', '1.5');
            svg.appendChild(line);
          });
        });
      }
    }
    if ('ResizeObserver' in window) { new ResizeObserver(draw).observe(stage); }
    else { window.addEventListener('resize', draw); }
    new MutationObserver(draw).observe(document.documentElement,
      { attributes: true, attributeFilter: ['data-theme', 'data-fs'] });
    requestAnimationFrame(draw);
  }

  /* ---------- ```viz type=highway：残差流主干 + 子层盒（取样/写回双线 + ⊕ 并入）---------- */
  function initHighway(el) {
    if (el.__plotInit) return;
    el.__plotInit = true;
    var cfg;
    try { cfg = JSON.parse(el.getAttribute('data-viz')); }
    catch (e) { el.textContent = '[viz 配置异常]'; return; }
    var KIND_CLS = { input: 'viz-k-input', op: 'viz-k-op', mid: 'viz-k-mid', output: 'viz-k-output', green: 'viz-k-green' };
    var KIND_LABEL = { input: '输入', op: '变换', mid: '中间结果', output: '输出', green: '计算' };
    var legendOv = cfg.legend || {};

    el.innerHTML = '';
    if (cfg.title || cfg.subtitle) {
      var head = document.createElement('div'); head.className = 'viz-head';
      var t = document.createElement('span'); t.className = 'viz-title'; t.textContent = cfg.title || '';
      head.appendChild(t);
      if (cfg.subtitle) { var s = document.createElement('span'); s.className = 'viz-sub'; s.textContent = cfg.subtitle; head.appendChild(s); }
      el.appendChild(head);
    }
    var stage = document.createElement('div'); stage.className = 'hw-stage';
    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'hw-svg');
    stage.appendChild(svg);
    var trunk = document.createElement('div'); trunk.className = 'hw-trunk';
    var from = document.createElement('span'); from.className = 'viz-node viz-k-input'; from.textContent = (cfg.trunk || {}).from || '输入';
    var spacer = document.createElement('span'); spacer.className = 'hw-mid';
    var to = document.createElement('span'); to.className = 'viz-node viz-k-output'; to.textContent = (cfg.trunk || {}).to || '输出';
    trunk.appendChild(from); trunk.appendChild(spacer); trunk.appendChild(to);
    stage.appendChild(trunk);
    var branches = document.createElement('div'); branches.className = 'hw-branches';
    var used = {};
    (cfg.branches || []).forEach(function (b) {
      var kind = b.kind || 'mid';
      used[kind] = 1;
      var slot = document.createElement('div'); slot.className = 'hw-slot';
      var box = document.createElement('div'); box.className = 'hw-block ' + kind;
      var title = document.createElement('div'); title.className = 'hw-btitle'; title.textContent = b.label;
      box.appendChild(title);
      if (b.note) { var sub = document.createElement('div'); sub.className = 'hw-bsub'; sub.textContent = b.note; box.appendChild(sub); }
      slot.appendChild(box);
      branches.appendChild(slot);
    });
    stage.appendChild(branches);
    if (cfg.note) { var fn = document.createElement('div'); fn.className = 'hw-foot'; fn.textContent = cfg.note; stage.appendChild(fn); }
    el.appendChild(stage);
    if (cfg.formula && cfg.formula.length) {
      var fw = document.createElement('div'); fw.className = 'hw-formula';
      cfg.formula.forEach(function (fch, i) {
        if (i) { var a = document.createElement('span'); a.className = 'hw-farrow'; a.textContent = '─►'; fw.appendChild(a); }
        var c = document.createElement('span'); c.className = 'hw-fchip'; c.textContent = fch; fw.appendChild(c);
      });
      el.appendChild(fw);
    }
    var legend = document.createElement('div'); legend.className = 'viz-legend';
    var i0 = document.createElement('span'); i0.className = 'cv-lg';
    i0.innerHTML = '<b class="hw-leg-add">⊕</b>' + (legendOv.add || '加法并入：x ← x + Block(x)');
    legend.appendChild(i0);
    var i1 = document.createElement('span'); i1.className = 'cv-lg';
    i1.innerHTML = '<i class="hw-leg-line"></i>' + (legendOv.trunk || '残差流主干（梯度直通）');
    legend.appendChild(i1);
    Object.keys(used).forEach(function (k) {
      var item = document.createElement('span'); item.className = 'cv-lg';
      item.innerHTML = '<i class="cv-sw ' + (KIND_CLS[k] || 'viz-k-mid') + '"></i>' + (legendOv[k] || KIND_LABEL[k] || k);
      legend.appendChild(item);
    });
    el.appendChild(legend);
    if (cfg.src) {
      var det = document.createElement('details'); det.className = 'viz-src';
      det.innerHTML = '<summary>原始文本</summary><pre></pre>';
      det.querySelector('pre').textContent = cfg.src;
      el.appendChild(det);
    }

    function draw() {
      svg.innerHTML = '';
      var st = stage.getBoundingClientRect();
      if (st.width < 10) return;
      svg.setAttribute('viewBox', '0 0 ' + st.width + ' ' + st.height);
      var cFg4 = cv('--fg4', '#818b98');
      var cWarn = cv('--warn', '#9a6700');
      var cCard = cv('--card', '#ffffff');
      var NS = 'http://www.w3.org/2000/svg';
      var f = from.getBoundingClientRect(), t2 = to.getBoundingClientRect();
      var x1 = f.right - st.left + 4, x2 = t2.left - st.left - 10;
      var y = f.top - st.top + f.height / 2;
      var tl = document.createElementNS(NS, 'line');
      tl.setAttribute('x1', x1); tl.setAttribute('y1', y); tl.setAttribute('x2', x2); tl.setAttribute('y2', y);
      tl.setAttribute('stroke', cFg4); tl.setAttribute('stroke-width', '2.5');
      svg.appendChild(tl);
      var ta = document.createElementNS(NS, 'path');
      ta.setAttribute('d', 'M ' + x2 + ' ' + (y - 5) + ' L ' + x2 + ' ' + (y + 5) + ' L ' + (x2 + 9) + ' ' + y + ' Z');
      ta.setAttribute('fill', cFg4);
      svg.appendChild(ta);
      Array.from(branches.querySelectorAll('.hw-block')).forEach(function (box) {
        var r = box.getBoundingClientRect();
        var cx = r.left - st.left + r.width / 2;
        var topY = r.top - st.top - 3;
        var jx1 = cx - 26, jx2 = cx + 26;
        function vline(x, yFrom, yTo, arrowAtBottom) {
          var l = document.createElementNS(NS, 'line');
          l.setAttribute('x1', x); l.setAttribute('y1', yFrom);
          l.setAttribute('x2', x); l.setAttribute('y2', yTo);
          l.setAttribute('stroke', cFg4); l.setAttribute('stroke-width', '1.8');
          svg.appendChild(l);
          var a = document.createElementNS(NS, 'path');
          var tipY = arrowAtBottom ? yTo - 1 : yFrom + 1;
          var baseY = arrowAtBottom ? yTo - 9 : yFrom + 9;
          a.setAttribute('d', 'M ' + (x - 4.5) + ' ' + baseY + ' L ' + (x + 4.5) + ' ' + baseY + ' L ' + x + ' ' + tipY + ' Z');
          a.setAttribute('fill', cFg4);
          svg.appendChild(a);
        }
        vline(jx1, y + 10, topY, true);
        vline(jx2, topY, y - 10, false);
        var c1 = document.createElementNS(NS, 'circle');
        c1.setAttribute('cx', cx); c1.setAttribute('cy', y); c1.setAttribute('r', '8');
        c1.setAttribute('fill', cCard); c1.setAttribute('stroke', cWarn); c1.setAttribute('stroke-width', '2');
        svg.appendChild(c1);
        var pl1 = document.createElementNS(NS, 'line');
        pl1.setAttribute('x1', cx - 4); pl1.setAttribute('y1', y); pl1.setAttribute('x2', cx + 4); pl1.setAttribute('y2', y);
        pl1.setAttribute('stroke', cWarn); pl1.setAttribute('stroke-width', '1.8');
        svg.appendChild(pl1);
        var pl2 = document.createElementNS(NS, 'line');
        pl2.setAttribute('x1', cx); pl2.setAttribute('y1', y - 4); pl2.setAttribute('x2', cx); pl2.setAttribute('y2', y + 4);
        pl2.setAttribute('stroke', cWarn); pl2.setAttribute('stroke-width', '1.8');
        svg.appendChild(pl2);
      });
    }
    if ('ResizeObserver' in window) { new ResizeObserver(draw).observe(stage); }
    else { window.addEventListener('resize', draw); }
    new MutationObserver(draw).observe(document.documentElement,
      { attributes: true, attributeFilter: ['data-theme', 'data-fs'] });
    requestAnimationFrame(draw);
  }
  function boot() {
    document.querySelectorAll('.inline-plot').forEach(initPlot);
    document.querySelectorAll('.chunk-viz').forEach(initChunk);
    document.querySelectorAll('.viz-tree').forEach(initTree);
    document.querySelectorAll('.viz-highway').forEach(initHighway);
  }
  if (document.readyState === 'loading') { document.addEventListener('DOMContentLoaded', boot); }
  else { boot(); }
})();
