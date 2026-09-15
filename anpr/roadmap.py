"""Owner: person 5. The schematic tracker map.

A hand-drawn street layout rather than a geographic one. Roads are polylines
in config.ROADS, cameras sit at a fraction along a road, and roads that touch
become junctions -- so a vehicle's route between two cameras is a shortest
path along actual roads, not a straight line between two dots.

Two pictures come out of here:

    draw_tracker_map()  every camera, every route, and what was seen where
    draw_route()        one vehicle, its path drawn with arrows

Adding a third street with two more cameras is four lines in config.py.
"""

import heapq
import math

import cv2
import numpy as np

from . import config

# ---------------------------------------------------------------- palette
# Dark console look: this is a tracking display, not a document.
BG = (22, 18, 14)            # near-black, faintly blue
ROAD_EDGE = (78, 64, 48)
ROAD_FILL = (48, 40, 32)
ROAD_CENTRE = (96, 84, 66)
CAM_RING = (255, 255, 255)
CAM_DOT = (60, 60, 235)      # BGR -> red
TEXT = (235, 235, 235)
DIM = (150, 140, 130)
ROUTE = (90, 200, 255)       # BGR -> amber, the tracked path
ROUTE_INFERRED = (150, 150, 150)
PANEL = (34, 28, 22)

MARGIN = 70
PANEL_W = 330


# ---------------------------------------------------------------------------
# Road network
# ---------------------------------------------------------------------------

def _point_on(polyline, fraction):
    """Point at `fraction` of the way along a polyline, by arc length."""
    segs = []
    total = 0.0
    for (x1, y1), (x2, y2) in zip(polyline, polyline[1:]):
        d = math.hypot(x2 - x1, y2 - y1)
        segs.append(((x1, y1), (x2, y2), d))
        total += d
    if total == 0:
        return tuple(polyline[0])

    want = max(0.0, min(1.0, fraction)) * total
    run = 0.0
    for (x1, y1), (x2, y2), d in segs:
        if run + d >= want and d > 0:
            k = (want - run) / d
            return (x1 + (x2 - x1) * k, y1 + (y2 - y1) * k)
        run += d
    return tuple(polyline[-1])


class RoadNetwork:
    """Roads plus cameras as one graph, so routes can be found along streets."""

    def __init__(self, roads=None, placement=None, tolerance=None):
        # A layout drawn in the dashboard wins over the config defaults, so
        # moving a camera never means editing Python.
        if roads is None or placement is None:
            from . import layout
            saved_roads, saved_cams = layout.as_network_args()
            roads = roads if roads is not None else saved_roads
            placement = placement if placement is not None else saved_cams
        self.roads = roads
        self.placement = placement
        self.tol = tolerance if tolerance is not None else config.JUNCTION_TOLERANCE

        self.nodes: list[tuple[float, float]] = []
        self.edges: dict[int, list[tuple[int, float]]] = {}
        self.cam_node: dict[str, int] = {}
        self._build()

    # -- construction ------------------------------------------------------
    def _node_for(self, point):
        """Reuse a nearby node so roads meeting at a corner share a junction."""
        for i, (nx, ny) in enumerate(self.nodes):
            if math.hypot(nx - point[0], ny - point[1]) <= self.tol:
                return i
        self.nodes.append((float(point[0]), float(point[1])))
        self.edges[len(self.nodes) - 1] = []
        return len(self.nodes) - 1

    def _connect(self, a, b):
        if a == b:
            return
        d = math.dist(self.nodes[a], self.nodes[b])
        if all(n != b for n, _ in self.edges[a]):
            self.edges[a].append((b, d))
            self.edges[b].append((a, d))

    @staticmethod
    def _project(pt, a, b):
        """Where `pt` falls on segment a-b: (distance, fraction along)."""
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return math.dist(pt, a), 0.0
        k = ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / length_sq
        k = max(0.0, min(1.0, k))
        foot = (ax + dx * k, ay + dy * k)
        return math.dist(pt, foot), k

    def _split_at_junctions(self, road, pts, all_vertices):
        """Insert a point where another road meets this one mid-segment.

        A road that starts partway along another (a T-junction) shares no
        vertex with it, so without this the two would never connect and no
        route could cross between them.
        """
        out = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            hits = []
            for other_road, vertex in all_vertices:
                if other_road == road:
                    continue
                d, k = self._project(vertex, a, b)
                if d <= self.tol and 1e-6 < k < 1 - 1e-6:
                    hits.append((k, vertex))
            for _, vertex in sorted(hits):
                out.append(tuple(vertex))
            out.append(b)
        return out

    def _build(self):
        all_vertices = [(name, (float(x), float(y)))
                        for name, points in self.roads.items() for x, y in points]

        # Cameras are inserted into their road's point list, so the graph runs
        # through them rather than past them.
        for road, points in self.roads.items():
            pts = [(float(x), float(y)) for x, y in points]
            pts = self._split_at_junctions(road, pts, all_vertices)
            extra = []
            for cam_id, (cam_road, frac) in self.placement.items():
                if cam_road == road:
                    extra.append((frac, cam_id, _point_on(pts, frac)))
            extra.sort()

            ordered = []
            for frac, cam_id, pt in extra:
                ordered.append((frac, cam_id, pt))

            # Walk the road, dropping camera points in at the right place.
            chain = []
            cum, total = 0.0, sum(math.dist(a, b) for a, b in zip(pts, pts[1:])) or 1.0
            inserted = 0
            chain.append((pts[0], None))
            for a, b in zip(pts, pts[1:]):
                seg = math.dist(a, b)
                while inserted < len(ordered) and ordered[inserted][0] * total <= cum + seg:
                    chain.append((ordered[inserted][2], ordered[inserted][1]))
                    inserted += 1
                cum += seg
                chain.append((b, None))
            while inserted < len(ordered):
                chain.append((ordered[inserted][2], ordered[inserted][1]))
                inserted += 1

            prev = None
            for pt, cam_id in chain:
                idx = self._node_for(pt)
                if cam_id:
                    self.cam_node[cam_id] = idx
                if prev is not None:
                    self._connect(prev, idx)
                prev = idx

    # -- queries -----------------------------------------------------------
    def camera_point(self, cam_id):
        idx = self.cam_node.get(cam_id)
        return self.nodes[idx] if idx is not None else None

    def route(self, cam_a, cam_b):
        """Shortest path along the roads, as a list of points. [] if none."""
        start, goal = self.cam_node.get(cam_a), self.cam_node.get(cam_b)
        if start is None or goal is None:
            return []

        dist = {start: 0.0}
        prev: dict[int, int] = {}
        queue = [(0.0, start)]
        seen = set()
        while queue:
            d, node = heapq.heappop(queue)
            if node in seen:
                continue
            seen.add(node)
            if node == goal:
                break
            for nxt, w in self.edges[node]:
                nd = d + w
                if nd < dist.get(nxt, float("inf")):
                    dist[nxt] = nd
                    prev[nxt] = node
                    heapq.heappush(queue, (nd, nxt))

        if goal not in dist:
            return []
        path, node = [goal], goal
        while node != start:
            node = prev[node]
            path.append(node)
        return [self.nodes[i] for i in reversed(path)]

    def bounds(self):
        xs = [p[0] for p in self.nodes] or [0]
        ys = [p[1] for p in self.nodes] or [0]
        return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _canvas(net, panel=True):
    x0, y0, x1, y1 = net.bounds()
    w = int(x1 - x0) + MARGIN * 2 + (PANEL_W if panel else 0)
    h = int(y1 - y0) + MARGIN * 2 + 90
    img = np.full((max(h, 360), max(w, 640), 3), BG, dtype=np.uint8)

    def to_px(p):
        return (int(p[0] - x0 + MARGIN), int(p[1] - y0 + MARGIN + 60))

    return img, to_px


def _draw_roads(img, net, to_px):
    """Roads as an outlined band, which is the double line of a hand sketch
    drawn properly -- the joins take care of themselves."""
    for points in net.roads.values():
        pts = np.array([to_px(p) for p in points], dtype=np.int32)
        cv2.polylines(img, [pts], False, ROAD_EDGE, 26, cv2.LINE_AA)
        cv2.polylines(img, [pts], False, ROAD_FILL, 20, cv2.LINE_AA)
        for a, b in zip(pts, pts[1:]):
            _dashes(img, tuple(a), tuple(b), ROAD_CENTRE, 1, dash=10, gap=12)


def _dashes(img, p1, p2, colour, thickness, dash=12, gap=9):
    dist = math.dist(p1, p2)
    if dist < 1:
        return
    n = int(dist // (dash + gap)) + 1
    for i in range(n):
        a = min((i * (dash + gap)) / dist, 1.0)
        b = min((i * (dash + gap) + dash) / dist, 1.0)
        cv2.line(img,
                 (int(p1[0] + (p2[0] - p1[0]) * a), int(p1[1] + (p2[1] - p1[1]) * a)),
                 (int(p1[0] + (p2[0] - p1[0]) * b), int(p1[1] + (p2[1] - p1[1]) * b)),
                 colour, thickness, cv2.LINE_AA)


def _draw_cameras(img, net, to_px, counts=None, active=None, right_limit=None,
                  offset=22):
    for cam_id in sorted(net.cam_node):
        pt = to_px(net.camera_point(cam_id))
        on = active is None or cam_id in active
        ring = CAM_RING if on else (110, 110, 110)
        cv2.circle(img, pt, 16, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, pt, 13, ring, 2, cv2.LINE_AA)
        cv2.circle(img, pt, 7, CAM_DOT if on else (90, 90, 90), -1, cv2.LINE_AA)

        label = cam_id
        if counts is not None:
            label = f"{cam_id}  {counts.get(cam_id, 0)} seen"

        # Put the label on whichever side leaves it on the canvas, so a camera
        # near the right edge does not write itself under the panel.
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        edge = right_limit if right_limit is not None else img.shape[1]
        lx = pt[0] + offset
        if lx + tw > edge - 10:
            lx = pt[0] - offset - tw
        cv2.putText(img, label, (max(6, lx), pt[1] + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, TEXT if on else DIM, 2, cv2.LINE_AA)


def _arrow_path(img, points, colour, thickness=4, spacing=64):
    """Draw a route and put arrowheads along it so direction is obvious."""
    pts = [(int(x), int(y)) for x, y in points]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, colour, thickness + 4, cv2.LINE_AA)
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, colour, thickness, cv2.LINE_AA)

    run = 0.0
    for a, b in zip(pts, pts[1:]):
        d = math.dist(a, b)
        if d < 1:
            continue
        step = spacing
        while run + step <= d:
            run += step
            k = run / d
            mid = (int(a[0] + (b[0] - a[0]) * k), int(a[1] + (b[1] - a[1]) * k))
            back = (int(mid[0] - (b[0] - a[0]) / d * 16),
                    int(mid[1] - (b[1] - a[1]) / d * 16))
            cv2.arrowedLine(img, back, mid, (255, 255, 255), thickness,
                            cv2.LINE_AA, tipLength=1.1)
            step = spacing
        run -= d
        run = max(run, 0.0)


def _header(img, title, subtitle=""):
    cv2.putText(img, title, (MARGIN - 20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.95, TEXT, 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(img, subtitle, (MARGIN - 20, 62), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, DIM, 1, cv2.LINE_AA)


def draw_tracker_map(sightings: list[dict], links: list[dict],
                     out_path: str = None) -> str:
    """The overview: the street layout, every camera, and what passed each one.

    sightings: rows from database.get_all_sightings()
    links:     rows from database.get_links()
    """
    out_path = out_path or str(config.ROAD_MAP_PNG)
    net = RoadNetwork()
    img, to_px = _canvas(net, panel=True)

    counts: dict[str, int] = {}
    for s in sightings:
        counts[s["cam_id"]] = counts.get(s["cam_id"], 0) + 1

    _draw_roads(img, net, to_px)

    by_id = {s["id"]: s for s in sightings}
    drawn = 0
    for link in links:
        a, b = by_id.get(link["a_id"]), by_id.get(link["b_id"])
        if not a or not b or a["cam_id"] == b["cam_id"]:
            continue
        pts = net.route(a["cam_id"], b["cam_id"])
        if len(pts) < 2:
            continue
        colour = ROUTE_INFERRED if link.get("method") == "inferred" else ROUTE
        _arrow_path(img, [to_px(p) for p in pts], colour, 3)
        drawn += 1

    _draw_cameras(img, net, to_px, counts=counts,
                  right_limit=img.shape[1] - PANEL_W)
    _header(img, "Vehicle tracker",
            f"{len(sightings)} sightings  |  {drawn} tracked journeys  |  "
            f"{len(net.cam_node)} camera points")
    _side_panel(img, sightings)
    cv2.imwrite(out_path, img)
    return out_path


def _side_panel(img, sightings):
    """Every vehicle seen, listed per camera, down the right-hand side.

    Space is split between the cameras rather than first-come, so a busy
    camera cannot push the others off the panel entirely. Plates that were
    actually read are listed first -- an unread vehicle is still counted, but
    a reader wants to see the identities.
    """
    h, w = img.shape[:2]
    x = w - PANEL_W
    cv2.rectangle(img, (x, 0), (w, h), PANEL, -1)
    cv2.line(img, (x, 0), (x, h), (70, 60, 50), 1, cv2.LINE_AA)
    cv2.putText(img, "SEEN AT EACH CAMERA", (x + 18, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, TEXT, 2, cv2.LINE_AA)

    per_cam: dict[str, list[dict]] = {}
    for s in sightings:
        per_cam.setdefault(s["cam_id"], []).append(s)
    if not per_cam:
        cv2.putText(img, "nothing seen yet", (x + 18, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM, 1, cv2.LINE_AA)
        return

    usable = h - 90
    per_block = usable // len(per_cam)
    rows_each = max(1, (per_block - 40) // 20)

    y = 78
    for cam_id in sorted(per_cam):
        rows = per_cam[cam_id]
        named = sorted([r for r in rows if r.get("plate")],
                       key=lambda r: (r.get("plate_status") != "clean", r["t_in"]))
        unnamed = len(rows) - len(named)

        cv2.putText(img, cam_id, (x + 18, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, CAM_DOT, 2, cv2.LINE_AA)
        cv2.putText(img, f"{len(rows)} vehicles", (x + 70, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, DIM, 1, cv2.LINE_AA)
        y += 24

        shown = 0
        for r in named[:rows_each]:
            clean = r.get("plate_status") == "clean"
            cv2.putText(img, r["plate"], (x + 30, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, TEXT if clean else DIM, 1, cv2.LINE_AA)
            cv2.putText(img, r.get("vehicle_type") or "?", (x + 200, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, DIM, 1, cv2.LINE_AA)
            y += 20
            shown += 1

        tail = []
        if len(named) > shown:
            tail.append(f"+{len(named) - shown} more read")
        if unnamed:
            tail.append(f"{unnamed} unread")
        if tail:
            cv2.putText(img, ", ".join(tail), (x + 30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, DIM, 1, cv2.LINE_AA)
            y += 20
        y += 14


def draw_route(plate: str, hops: list[dict], out_path: str) -> str:
    """One vehicle's journey: its path drawn with arrows, cameras it missed dimmed.

    hops: the sightings for this vehicle, in time order, each with cam_id and
    t_in. Consecutive pairs are drawn as arrows along the roads between them.
    """
    net = RoadNetwork()
    img, to_px = _canvas(net, panel=False)
    _draw_roads(img, net, to_px)

    visited = [h["cam_id"] for h in hops]
    for a, b in zip(hops, hops[1:]):
        pts = net.route(a["cam_id"], b["cam_id"])
        if len(pts) >= 2:
            _arrow_path(img, [to_px(p) for p in pts], ROUTE, 4)

    # 38 clears the halo drawn round a visited stop
    if len(set(visited)) == 1:
        pt = net.camera_point(visited[0])
        if pt is not None:
            px = to_px(pt)
            for r, shade in ((34, (46, 40, 32)), (26, (60, 52, 40))):
                cv2.circle(img, px, r, shade, 2, cv2.LINE_AA)

    _draw_cameras(img, net, to_px, active=set(visited), offset=38)

    # Stamp each stop with when the vehicle was there.
    base = hops[0]["t_in"] if hops else 0.0
    for i, h in enumerate(hops):
        pt = net.camera_point(h["cam_id"])
        if pt is None:
            continue
        px = to_px(pt)
        cv2.putText(img, f"{i + 1}", (px[0] - 5, px[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        stamp = f"t+{h['t_in'] - base:.1f}s"
        (sw, sh), sb = cv2.getTextSize(stamp, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        sx, sy = px[0] + 38, px[1] + 30
        sx = max(6, min(sx, img.shape[1] - sw - 12))
        sy = max(sh + sb + 8, min(sy, img.shape[0] - 8))
        # A stop can sit anywhere, including on top of a road, so give the
        # timestamp its own backing rather than hoping the spot is empty.
        cv2.rectangle(img, (sx - 5, sy - sh - sb - 4), (sx + sw + 5, sy + 4),
                      (18, 14, 10), -1)
        cv2.putText(img, stamp, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, ROUTE, 1, cv2.LINE_AA)

    # Say plainly what this card is showing. A vehicle seen once is a real
    # result, not a failure -- claiming a route it never took would be worse.
    unique_cams = []
    for cam in visited:
        if cam not in unique_cams:
            unique_cams.append(cam)

    if len(unique_cams) >= 2:
        span = hops[-1]["t_in"] - hops[0]["t_in"]
        route_text = "  ->  ".join(unique_cams) + f"    tracked across {span:.0f}s"
    elif unique_cams:
        seen_type = hops[0].get("vehicle_type") or "vehicle"
        colour = hops[0].get("colour")
        desc = f"{colour} {seen_type}" if colour else seen_type
        route_text = (f"seen at {unique_cams[0]} only  |  {desc}  |  "
                      f"no second camera has read this plate")
    else:
        route_text = "no camera position for this sighting"
    _header(img, f"Tracking  {plate}", route_text)

    cv2.imwrite(out_path, img)
    return out_path


# ---------------------------------------------------------------------------
# Layout preview: see config.ROADS without running the pipeline
# ---------------------------------------------------------------------------

def draw_layout_preview(out_path: str = None, grid: bool = True) -> str:
    """Render the road layout on its own, with a coordinate grid.

    Editing ROADS is guesswork without seeing it, and a full pipeline run
    takes minutes. This draws just the map, in about a second, with the grid
    and every point labelled so the next number to type is readable straight
    off the picture.

        python -m anpr.roadmap
    """
    out_path = out_path or str(config.OUT_DIR / "layout_preview.png")
    net = RoadNetwork()
    img, to_px = _canvas(net, panel=False)

    if grid:
        x0, y0, x1, y1 = net.bounds()
        step = 50
        gx = int(x0 // step) * step
        while gx <= x1 + step:
            px = to_px((gx, y0))[0]
            cv2.line(img, (px, 0), (px, img.shape[0]), (42, 36, 30), 1)
            cv2.putText(img, str(gx), (px + 3, 78), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (110, 100, 92), 1, cv2.LINE_AA)
            gx += step
        gy = int(y0 // step) * step
        while gy <= y1 + step:
            py = to_px((x0, gy))[1]
            cv2.line(img, (0, py), (img.shape[1], py), (42, 36, 30), 1)
            cv2.putText(img, str(gy), (6, py - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (110, 100, 92), 1, cv2.LINE_AA)
            gy += step

    _draw_roads(img, net, to_px)

    # Label every road point with the exact numbers that produced it.
    for name, points in net.roads.items():
        for x, y in points:
            px = to_px((x, y))
            cv2.circle(img, px, 4, (120, 200, 120), -1, cv2.LINE_AA)
            cv2.putText(img, f"({int(x)},{int(y)})", (px[0] + 7, px[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 200, 120), 1, cv2.LINE_AA)
        mid = to_px(_point_on([(float(a), float(b)) for a, b in points], 0.5))
        cv2.putText(img, name, (mid[0] + 10, mid[1] + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (150, 170, 200), 1, cv2.LINE_AA)

    # Every camera, and a route between each consecutive pair, so a broken
    # junction shows up here rather than after a ten minute run.
    cams = sorted(net.cam_node)
    for a, b in zip(cams, cams[1:]):
        pts = net.route(a, b)
        if len(pts) >= 2:
            _arrow_path(img, [to_px(p) for p in pts], ROUTE, 3)
    _draw_cameras(img, net, to_px)

    for cam_id, (road, frac) in sorted(config.CAMERA_PLACEMENT.items()):
        pt = net.camera_point(cam_id)
        if pt:
            px = to_px(pt)
            cv2.putText(img, f'("{road}", {frac})', (px[0] + 22, px[1] + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, DIM, 1, cv2.LINE_AA)

    broken = [f"{a}-{b}" for a in cams for b in cams
              if a < b and len(net.route(a, b)) < 2]
    sub = f"{len(net.roads)} roads  |  {len(cams)} cameras  |  {len(net.nodes)} nodes"
    if broken:
        sub += f"  |  NOT CONNECTED: {', '.join(broken)}"
    _header(img, "Road layout preview", sub)

    cv2.imwrite(out_path, img)
    return out_path


if __name__ == "__main__":
    print(draw_layout_preview())
