/* ANPR dashboard.
   Left: draw the streets, drop cameras, assign clips, run.
   Right: every vehicle the run identified, its path and its evidence. */

(function () {
  'use strict';

  var W = 960, H = 620;              // editor coordinate space
  var state = {
    roads: {},                       // name -> [[x,y], ...]
    cameras: {},                     // cam -> [road, fraction]
    drawing: null,                   // points of the road being drawn
    tool: 'road',
    clips: [],
    assign: {},                      // cam -> clip filename
    offsets: {},                     // cam -> seconds after the first clip
    results: null,
    dragging: null,
    tab: 'vehicles',
    filter: 'all',
    startedAt: null,
    elapsedTimer: null
  };

  var $ = function (id) { return document.getElementById(id); };
  var canvas, ctx;

  function tok(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  // ------------------------------------------------------------ geometry
  function pointOn(points, frac) {
    var segs = [], total = 0, i;
    for (i = 0; i < points.length - 1; i++) {
      var d = Math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1]);
      segs.push(d); total += d;
    }
    if (!total) return points[0].slice();
    var want = Math.max(0, Math.min(1, frac)) * total, run = 0;
    for (i = 0; i < segs.length; i++) {
      if (run + segs[i] >= want && segs[i] > 0) {
        var k = (want - run) / segs[i];
        return [points[i][0] + (points[i + 1][0] - points[i][0]) * k,
                points[i][1] + (points[i + 1][1] - points[i][1]) * k];
      }
      run += segs[i];
    }
    return points[points.length - 1].slice();
  }

  /* Nearest point on a road, and how far along it — this is what turns a
     click anywhere near a line into a camera placed ON that line. */
  function projectToRoad(points, p) {
    var best = { dist: Infinity, frac: 0 }, total = 0, lens = [], i;
    for (i = 0; i < points.length - 1; i++) {
      var d = Math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1]);
      lens.push(d); total += d;
    }
    if (!total) return best;
    var run = 0;
    for (i = 0; i < points.length - 1; i++) {
      var a = points[i], b = points[i + 1];
      var dx = b[0] - a[0], dy = b[1] - a[1], L = dx * dx + dy * dy;
      var k = L ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L : 0;
      k = Math.max(0, Math.min(1, k));
      var fx = a[0] + dx * k, fy = a[1] + dy * k;
      var dist = Math.hypot(p[0] - fx, p[1] - fy);
      if (dist < best.dist) best = { dist: dist, frac: (run + lens[i] * k) / total };
      run += lens[i];
    }
    return best;
  }

  function nearestRoad(p) {
    var best = null;
    Object.keys(state.roads).forEach(function (name) {
      var r = projectToRoad(state.roads[name], p);
      if (!best || r.dist < best.dist) best = { road: name, frac: r.frac, dist: r.dist };
    });
    return best;
  }

  function nextCamName() {
    for (var i = 1; i < 100; i++) {
      if (!state.cameras['C' + i]) return 'C' + i;
    }
    return 'C99';
  }

  // --------------------------------------------------------------- canvas
  function draw() {
    if (!ctx) return;
    var line = tok('--line'), ink = tok('--ink'), dim = tok('--ink-dim');
    var cyan = tok('--cyan'), surface2 = tok('--surface-2'), sun = tok('--sun');

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = surface2;
    ctx.fillRect(0, 0, W, H);

    // grid
    ctx.strokeStyle = line; ctx.globalAlpha = .35; ctx.lineWidth = 1;
    for (var g = 0; g <= W; g += 40) {
      ctx.beginPath(); ctx.moveTo(g, 0); ctx.lineTo(g, H); ctx.stroke();
    }
    for (var gy = 0; gy <= H; gy += 40) {
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // roads: a wide band with a dashed centre line, so they read as streets
    Object.keys(state.roads).forEach(function (name) {
      var pts = state.roads[name];
      strokePath(pts, line, 26);
      strokePath(pts, surface2, 19);
      ctx.save();
      ctx.setLineDash([9, 11]);
      strokePath(pts, dim, 1.5);
      ctx.restore();

      pts.forEach(function (p) {
        ctx.fillStyle = line;
        ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, Math.PI * 2); ctx.fill();
      });

      var mid = pointOn(pts, .5);
      ctx.fillStyle = dim;
      ctx.font = '12px "Spline Sans Mono", monospace';
      ctx.fillText(name, mid[0] + 12, mid[1] + 22);
    });

    // the road being drawn right now
    if (state.drawing && state.drawing.length) {
      ctx.save();
      ctx.setLineDash([7, 7]);
      strokePath(state.drawing, cyan, 2.5);
      ctx.restore();
      state.drawing.forEach(function (p) {
        ctx.fillStyle = cyan;
        ctx.beginPath(); ctx.arc(p[0], p[1], 5, 0, Math.PI * 2); ctx.fill();
      });
    }

    // cameras
    Object.keys(state.cameras).forEach(function (cam) {
      var place = state.cameras[cam];
      var pts = state.roads[place[0]];
      if (!pts) return;
      var p = pointOn(pts, place[1]);
      var assigned = !!state.assign[cam];

      ctx.fillStyle = surface2;
      ctx.beginPath(); ctx.arc(p[0], p[1], 13, 0, Math.PI * 2); ctx.fill();
      ctx.lineWidth = 2.5; ctx.strokeStyle = assigned ? cyan : dim;
      ctx.beginPath(); ctx.arc(p[0], p[1], 13, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = assigned ? cyan : dim;
      ctx.beginPath(); ctx.arc(p[0], p[1], 6, 0, Math.PI * 2); ctx.fill();

      ctx.fillStyle = ink;
      ctx.font = '600 14px "Spline Sans Mono", monospace';
      ctx.fillText(cam, p[0] + 19, p[1] + 5);

      var seen = seenCount(cam);
      if (seen !== null) {
        ctx.fillStyle = sun;
        ctx.font = '11px "Spline Sans Mono", monospace';
        ctx.fillText(seen + ' seen', p[0] + 19, p[1] + 21);
      }
    });
  }

  function strokePath(pts, colour, width) {
    if (pts.length < 2) return;
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();
  }

  function seenCount(cam) {
    if (!state.results || !state.results.cameras) return null;
    var row = state.results.cameras.filter(function (c) { return c.cam_id === cam; })[0];
    return row ? row.seen : null;
  }

  function canvasPoint(evt) {
    var r = canvas.getBoundingClientRect();
    return [Math.round((evt.clientX - r.left) / r.width * W),
            Math.round((evt.clientY - r.top) / r.height * H)];
  }

  // ----------------------------------------------------------- editing
  function onCanvasDown(evt) {
    var p = canvasPoint(evt);

    if (state.tool === 'road') {
      if (!state.drawing) state.drawing = [];
      state.drawing.push(p);
      draw();
      return;
    }

    if (state.tool === 'camera') {
      var near = nearestRoad(p);
      if (!near || near.dist > 40) {
        setLayoutMsg('click closer to a road — cameras sit on streets', 'failed');
        return;
      }
      state.cameras[nextCamName()] = [near.road, near.frac];
      renderCameras(); draw();
      setLayoutMsg('camera placed — save when you are happy', '');
      return;
    }

    if (state.tool === 'erase') {
      var hit = cameraAt(p);
      if (hit) {
        delete state.cameras[hit];
        delete state.assign[hit];
        renderCameras(); draw();
        return;
      }
      var near2 = nearestRoad(p);
      if (near2 && near2.dist < 22) {
        var name = near2.road;
        Object.keys(state.cameras).forEach(function (cam) {
          if (state.cameras[cam][0] === name) { delete state.cameras[cam]; delete state.assign[cam]; }
        });
        delete state.roads[name];
        renderCameras(); draw();
      }
      return;
    }

    if (state.tool === 'move') {
      var cam = cameraAt(p);
      if (cam) { state.dragging = { kind: 'camera', cam: cam }; return; }
      var best = null;
      Object.keys(state.roads).forEach(function (name) {
        state.roads[name].forEach(function (pt, i) {
          var d = Math.hypot(pt[0] - p[0], pt[1] - p[1]);
          if (d < 16 && (!best || d < best.d)) best = { kind: 'point', road: name, i: i, d: d };
        });
      });
      if (best) state.dragging = best;
    }
  }

  function onCanvasMove(evt) {
    if (!state.dragging) return;
    var p = canvasPoint(evt);
    if (state.dragging.kind === 'camera') {
      var near = nearestRoad(p);
      if (near) state.cameras[state.dragging.cam] = [near.road, near.frac];
    } else {
      state.roads[state.dragging.road][state.dragging.i] = p;
    }
    draw();
  }

  function onCanvasUp() { state.dragging = null; }

  function cameraAt(p) {
    var found = null;
    Object.keys(state.cameras).forEach(function (cam) {
      var pts = state.roads[state.cameras[cam][0]];
      if (!pts) return;
      var c = pointOn(pts, state.cameras[cam][1]);
      if (Math.hypot(c[0] - p[0], c[1] - p[1]) < 18) found = cam;
    });
    return found;
  }

  function finishRoad() {
    if (!state.drawing || state.drawing.length < 2) {
      setLayoutMsg('a road needs at least two points', 'failed');
      state.drawing = null; draw();
      return;
    }
    var n = 1;
    while (state.roads['street_' + n]) n++;
    state.roads['street_' + n] = state.drawing;
    state.drawing = null;
    draw();
    setLayoutMsg('street_' + n + ' added', '');
  }

  var TOOL_NOTES = {
    road: 'click to add road points, then Finish road',
    camera: 'click on a road to drop a camera there',
    move: 'drag a camera or a road point',
    erase: 'click a camera to remove it, or a road to delete the whole street'
  };

  function setTool(name) {
    state.tool = name;
    var note = document.getElementById('canvasNote');
    if (note) note.textContent = TOOL_NOTES[name] || '';
    [['road', 'toolRoad'], ['camera', 'toolCamera'],
     ['move', 'toolMove'], ['erase', 'toolErase']].forEach(function (pair) {
      $(pair[1]).classList.toggle('active', state.tool === pair[0]);
    });
    if (name !== 'road' && state.drawing && state.drawing.length >= 2) finishRoad();
  }

  // -------------------------------------------------------------- server
  function api(url, opts) {
    return fetch(url, opts).then(function (r) {
      return r.json().then(function (body) {
        if (!r.ok) throw new Error(body.error || ('HTTP ' + r.status));
        return body;
      });
    });
  }

  function setLayoutMsg(msg, cls) {
    $('layoutMsg').textContent = msg;
    $('layoutDot').className = 'dot' + (cls ? ' ' + cls : '');
  }

  function loadLayout() {
    return api('/api/layout').then(function (d) {
      state.roads = d.roads || {};
      state.cameras = d.cameras || {};
      renderCameras(); draw();
      setLayoutMsg(d.saved ? 'loaded your saved layout' : 'showing the default layout', '');
    });
  }

  function saveLayout() {
    if (state.drawing && state.drawing.length >= 2) finishRoad();
    api('/api/layout', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ roads: state.roads, cameras: state.cameras })
    }).then(function () {
      setLayoutMsg('layout saved', 'done');
    }).catch(function (e) {
      setLayoutMsg(e.message, 'failed');
    });
  }

  function loadClips() {
    return api('/api/clips').then(function (d) {
      state.clips = d.clips || [];
      renderClips(); renderCameras();
    });
  }

  function renderClips() {
    var ul = $('clipList');
    ul.innerHTML = '';
    if (!state.clips.length) {
      ul.innerHTML = '<li class="text-dim" style="font-size:13px">' +
        'No clips yet. Upload one above, or drop files into anpr/data/videos.</li>';
      return;
    }
    state.clips.forEach(function (c) {
      var li = document.createElement('li');
      var meta = [c.seconds ? c.seconds + 's' : null,
                  c.fps ? c.fps + ' fps' : null,
                  c.width ? c.width + '×' + c.height : null,
                  c.size_mb + ' MB'].filter(Boolean).join('  ·  ');
      li.innerHTML = '<div class="clip-name"><b></b><span class="clip-meta"></span></div>';
      li.querySelector('b').textContent = c.name;
      li.querySelector('.clip-meta').textContent = meta;
      ul.appendChild(li);
    });
  }

  function renderCameras() {
    var ul = $('camList');
    ul.innerHTML = '';
    var cams = Object.keys(state.cameras).sort();
    if (!cams.length) {
      ul.innerHTML = '<li class="text-dim" style="font-size:13px">' +
        'No cameras yet. Use "Place camera" and click on a road.</li>';
      return;
    }
    cams.forEach(function (cam) {
      var li = document.createElement('li');
      var tag = document.createElement('span');
      tag.className = 'cam-tag';
      tag.textContent = cam;

      var sel = document.createElement('select');
      sel.innerHTML = '<option value="">— no clip —</option>';
      state.clips.forEach(function (c) {
        var o = document.createElement('option');
        o.value = c.name; o.textContent = c.name;
        if (state.assign[cam] === c.name) o.selected = true;
        sel.appendChild(o);
      });
      sel.addEventListener('change', function () {
        if (sel.value) state.assign[cam] = sel.value; else delete state.assign[cam];
        draw();
      });

      // Seconds after the first clip. Only the differences matter, so an
      // offset is easier to supply than a wall-clock time and says the same.
      var off = document.createElement('input');
      off.className = 'input offset';
      off.type = 'number';
      off.min = '0';
      off.placeholder = '+0s';
      off.title = 'seconds after the first clip started recording';
      off.value = state.offsets[cam] != null ? state.offsets[cam] : '';
      off.addEventListener('change', function () {
        var v = parseFloat(off.value);
        if (isNaN(v)) delete state.offsets[cam]; else state.offsets[cam] = v;
      });

      li.appendChild(tag);
      li.appendChild(sel);
      li.appendChild(off);
      ul.appendChild(li);
    });
  }

  function upload(files) {
    var queue = Array.prototype.slice.call(files);
    if (!queue.length) return;
    setLayoutMsg('uploading ' + queue.length + ' file(s)…', 'running');
    Promise.all(queue.map(function (f) {
      var fd = new FormData();
      fd.append('file', f);
      return fetch('/api/clips', { method: 'POST', body: fd });
    })).then(loadClips).then(function () {
      setLayoutMsg('clips added', 'done');
    });
  }

  // ----------------------------------------------------------- the run
  function startRun() {
    if (!Object.keys(state.assign).length) {
      setRunMsg('assign a clip to at least one camera first', 'failed');
      return;
    }
    $('runError').classList.add('hidden');
    $('runBtn').disabled = true;
    var seconds = $('secondsInput').value;

    api('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sources: state.assign, seconds: seconds || null,
                             starts: state.offsets })
    }).then(function () {
      setRunMsg('starting…', 'running');
      showProcessing(true);
      poll();
    }).catch(function (e) {
      $('runBtn').disabled = false;
      setRunMsg(e.message, 'failed');
    });
  }

  function setRunMsg(msg, cls) {
    $('runMsg').textContent = msg;
    $('runDot').className = 'dot' + (cls ? ' ' + cls : '');
  }

  /* A Python without the pipeline's packages produces a run with zero
     vehicles and no error, which looks like bad footage. Say so plainly. */
  function warnIfBroken(s) {
    if (!s.missing || !s.missing.length) return false;
    var names = s.missing.map(function (m) { return m.module; }).join(', ');
    setRunMsg('cannot run: missing ' + names, 'failed');
    $('runError').textContent =
      ['This Python is missing: ' + names + '.',
       '',
       'Stop the server and start it with the project venv:',
       '    .venv\\Scripts\\python.exe -m webapp.app'].join('\n');
    $('runError').classList.remove('hidden');
    $('runBtn').disabled = true;
    return true;
  }

  function poll() {
    api('/api/status').then(function (s) {
      if (warnIfBroken(s)) { showProcessing(false); return; }
      if (s.status === 'running') {
        setRunMsg(s.message || 'working…', 'running');
        $('procMsg').textContent = s.message || 'working…';
        setTimeout(poll, 1200);
        return;
      }
      $('runBtn').disabled = false;
      showProcessing(false);
      if (s.status === 'failed') {
        setRunMsg(s.error || 'failed', 'failed');
        $('runError').textContent = s.message || '';
        $('runError').classList.remove('hidden');
        return;
      }
      setRunMsg('finished', 'done');
      loadResults();
    });
  }

  /* While a run is going, the results area shows what it is doing instead of
     the previous run's vehicles -- stale cards next to a spinning status read
     as if the new run had already finished. */
  function showProcessing(on) {
    $('tabProcessing').classList.toggle('hidden', !on);
    ['tabVehicles', 'tabMap', 'tabCameras'].forEach(function (id) {
      $(id).classList.toggle('hidden', on || id !== 'tab' + cap(state.tab));
    });
    $('statRow').classList.toggle('hidden', on);
    document.querySelectorAll('[data-tab]').forEach(function (b) { b.disabled = on; });
    $('resultTitle').textContent = on ? 'Processing' : 'Results';

    clearInterval(state.elapsedTimer);
    if (on) {
      state.startedAt = Date.now();
      $('procElapsed').textContent = '0s elapsed';
      state.elapsedTimer = setInterval(function () {
        var secs = Math.round((Date.now() - state.startedAt) / 1000);
        var mins = Math.floor(secs / 60);
        $('procElapsed').textContent =
          (mins ? mins + 'm ' + (secs % 60) + 's' : secs + 's') + ' elapsed';
      }, 1000);
    }
  }

  function cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

  // --------------------------------------------------------- results
  function loadResults() {
    return api('/api/results').then(function (d) {
      state.results = d;
      renderResults();
      draw();
    });
  }

  /* One card per vehicle the cameras tracked, identified or not. The filter
     decides which of them are on screen; the counts next to it are what make
     "29 seen, 5 identified" make sense at a glance. */
  function renderVehicles() {
    var d = state.results || {};
    var all = d.vehicles || [];
    var identified = all.filter(function (v) { return v.plate; });
    var tracked = all.filter(function (v) { return v.linked; });

    var counts = { all: all.length, identified: identified.length, tracked: tracked.length };
    document.querySelectorAll('[data-filter]').forEach(function (b) {
      var k = b.getAttribute('data-filter');
      b.textContent = b.getAttribute('data-label') + '  ' + counts[k];
      b.classList.toggle('active', k === state.filter);
    });

    var vehicles = state.filter === 'identified' ? identified
                 : state.filter === 'tracked' ? tracked : all;

    $('filterNote').textContent = state.filter === 'all' && d.unread
      ? d.unread + ' of these had no readable plate'
      : '';

    var grid = $('vehicleGrid');
    grid.innerHTML = '';
    $('vehicleEmpty').classList.toggle('hidden', vehicles.length > 0);

    vehicles.forEach(function (v) {
      var card = document.createElement('button');
      card.className = 'vcard';
      card.type = 'button';

      var thumb = (v.hops[0] && v.hops[0].crop) || v.journey_img;
      var cams = Array.from(new Set(v.cameras));
      card.innerHTML =
        '<img class="thumb" alt="">' +
        '<div class="body">' +
          '<p class="eyebrow"></p>' +
          '<h3 class="card-title"></h3>' +
          '<div class="vmeta"><span class="desc"></span><span class="badge"></span></div>' +
        '</div>';

      var img = card.querySelector('.thumb');
      if (thumb) { img.src = thumb; img.alt = 'Vehicle ' + v.plate; }
      else { img.removeAttribute('src'); img.alt = ''; }

      card.querySelector('.eyebrow').textContent = cams.join(' → ');
      card.querySelector('.card-title').textContent = v.plate || 'no plate read';
      if (!v.plate) card.classList.add('unread');
      card.querySelector('.desc').textContent =
        [v.colour, v.type].filter(Boolean).join(' ') || 'vehicle';

      var badge = card.querySelector('.badge');
      if (v.linked) { badge.textContent = 'tracked'; badge.className = 'badge tracked'; }
      else if (v.status === 'clean') { badge.textContent = 'clean'; badge.className = 'badge clean'; }
      else { badge.textContent = v.status || 'partial'; }

      card.addEventListener('click', function () { openVehicle(v); });
      grid.appendChild(card);
    });
  }


  function renderResults() {
    var d = state.results;
    // The list now holds unidentified vehicles too, so "identified" has to
    // count the ones with a plate rather than the length of the list.
    var identified = (d.vehicles || []).filter(function (v) { return v.plate; }).length;
    var tracked = (d.vehicles || []).filter(function (v) { return v.linked; }).length;

    $('navStats').innerHTML = '';
    [['sightings', d.sightings || 0], ['identified', identified],
     ['tracked', tracked]].forEach(function (pair) {
      var s = document.createElement('span');
      s.innerHTML = pair[0] + ' <b></b>';
      s.querySelector('b').textContent = pair[1];
      $('navStats').appendChild(s);
    });

    var stats = [
      ['vehicles seen', d.sightings || 0, false],
      ['identified', identified, false],
      ['tracked across cameras', tracked, true],
      ['no plate read', d.unread || 0, false]
    ];
    $('statRow').innerHTML = '';
    stats.forEach(function (s) {
      var el = document.createElement('div');
      el.className = 'stat';
      el.innerHTML = '<span class="n' + (s[2] ? ' value' : '') + '"></span>' +
                     '<span class="k"></span>';
      el.querySelector('.n').textContent = s[1];
      el.querySelector('.k').textContent = s[0];
      $('statRow').appendChild(el);
    });

    renderVehicles();

    var map = $('trackerMap');
    if (d.tracker_map) { map.src = d.tracker_map + '?t=' + Date.now(); map.classList.remove('hidden'); }
    else { map.classList.add('hidden'); }

    var mg = $('montageGrid');
    mg.innerHTML = '';
    (d.montages || []).forEach(function (src) {
      var b = document.createElement('button');
      b.type = 'button';
      b.innerHTML = '<img alt="All cameras at one moment">';
      b.querySelector('img').src = src + '?t=' + Date.now();
      b.addEventListener('click', function () { window.open(src, '_blank'); });
      mg.appendChild(b);
    });

    $('resultTitle').textContent = d.ran ? 'Results' : 'Results';
  }

  function openVehicle(v) {
    $('modalPlate').textContent = v.plate;
    $('modalRoute').textContent = Array.from(new Set(v.cameras)).join('  →  ');
    $('modalDesc').textContent =
      [v.colour, v.type].filter(Boolean).join(' ') +
      (v.linked ? ' · tracked across cameras'
                : ' · seen at ' + Array.from(new Set(v.cameras)).length + ' camera');

    var conf = $('modalConf');
    conf.innerHTML = '';
    var first = v.hops[0];
    if (first && first.conf && first.conf.length) {
      v.plate.split('').forEach(function (ch, i) {
        var c = first.conf[i];
        if (typeof c !== 'number') return;
        var el = document.createElement('span');
        el.className = 'conf ' + (c >= 0.8 ? 'hi' : c >= 0.5 ? '' : 'lo');
        el.innerHTML = '<b></b><span></span>';
        el.querySelector('b').textContent = ch;
        el.querySelector('span').textContent = Math.round(c * 100);
        conf.appendChild(el);
      });
    }

    setImg($('modalRouteImg'), v.route_img);
    setImg($('modalJourneyImg'), v.journey_img);

    var hops = $('modalHops');
    hops.innerHTML = '';
    v.hops.forEach(function (h, i) {
      var li = document.createElement('li');
      li.innerHTML = '<img alt=""><div style="flex:1">' +
        '<b class="stat-mono"></b> <span class="text-dim"></span></div>' +
        '<span class="price"></span>';
      var im = li.querySelector('img');
      if (h.crop) { im.src = h.crop; im.alt = 'Seen at ' + h.cam_id; }
      else im.remove();
      li.querySelector('b').textContent = (i + 1) + '.  ' + h.cam_id;
      li.querySelector('.text-dim').textContent = h.direction ? 'heading ' + h.direction : '';
      li.querySelector('.price').textContent = 't+' + h.t + 's';
      hops.appendChild(li);
    });

    $('overlay').classList.remove('hidden');
  }

  function setImg(el, src) {
    if (src) { el.src = src + '?t=' + Date.now(); el.classList.remove('hidden'); }
    else { el.removeAttribute('src'); el.classList.add('hidden'); }
  }

  // ------------------------------------------------------------- wiring
  document.addEventListener('DOMContentLoaded', function () {
    canvas = $('editor');
    if (!canvas) return;
    canvas.width = W; canvas.height = H;
    ctx = canvas.getContext('2d');

    canvas.addEventListener('pointerdown', onCanvasDown);
    canvas.addEventListener('pointermove', onCanvasMove);
    window.addEventListener('pointerup', onCanvasUp);

    $('toolRoad').addEventListener('click', function () { setTool('road'); });
    $('toolCamera').addEventListener('click', function () { setTool('camera'); });
    $('toolMove').addEventListener('click', function () { setTool('move'); });
    $('toolErase').addEventListener('click', function () { setTool('erase'); });
    $('finishRoad').addEventListener('click', finishRoad);

    $('clearAll').addEventListener('click', function () {
      state.roads = {}; state.cameras = {}; state.assign = {}; state.drawing = null;
      renderCameras(); draw();
      setLayoutMsg('cleared — draw a road to start', '');
    });

    $('saveLayout').addEventListener('click', saveLayout);
    $('resetLayout').addEventListener('click', function () {
      api('/api/layout/reset', { method: 'POST' }).then(function (d) {
        state.roads = d.roads; state.cameras = d.cameras;
        renderCameras(); draw();
        setLayoutMsg('back to the default layout', '');
      });
    });

    $('fileInput').addEventListener('change', function (e) { upload(e.target.files); });
    $('runBtn').addEventListener('click', startRun);

    document.querySelectorAll('[data-filter]').forEach(function (btn) {
      btn.setAttribute('data-label', btn.textContent.trim());
      btn.addEventListener('click', function () {
        state.filter = btn.getAttribute('data-filter');
        renderVehicles();
      });
    });

    document.querySelectorAll('[data-tab]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        document.querySelectorAll('[data-tab]').forEach(function (b) {
          b.classList.toggle('active', b === btn);
        });
        state.tab = btn.getAttribute('data-tab');
        $('tabVehicles').classList.toggle('hidden', state.tab !== 'vehicles');
        $('tabMap').classList.toggle('hidden', state.tab !== 'map');
        $('tabCameras').classList.toggle('hidden', state.tab !== 'cameras');
      });
    });

    $('modalClose').addEventListener('click', function () {
      $('overlay').classList.add('hidden');
    });
    $('overlay').addEventListener('click', function (e) {
      if (e.target === $('overlay')) $('overlay').classList.add('hidden');
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') $('overlay').classList.add('hidden');
    });

    window.ANPR = { redraw: draw };

    api('/api/status').then(warnIfBroken);

    loadLayout().then(loadClips).then(loadResults).catch(function (e) {
      setLayoutMsg('could not reach the server: ' + e.message, 'failed');
    });
  });
})();
