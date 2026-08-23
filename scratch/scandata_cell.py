class ScanData:
    """
    Fast reader for CR-39 CPSA binary scan files.

    Parameters
    ----------
    path       : str   – Path to the .cpsa file.
    d_bounds   : (min, max) – Diameter filter in µm.
    e_bounds   : (min, max) – Eccentricity filter (raw int8).
    c_bounds   : (min, max) – Normal contrast filter (raw int8).
    a_bounds   : (min, max) – Average contrast filter (raw int8).
    x_bounds   : (min, max) – Spatial x filter in cm.
    y_bounds   : (min, max) – Spatial y filter in cm.

    Attributes
    ----------
    header  : dict       – Scan metadata (pixel size, frame dimensions, …)
    frames  : DataFrame  – Per-frame info (position, num_tracks, focus, …)
    tracks  : DataFrame  – Track data (d, x, y, e, c, a, frame_number)
    trailer : str        – ASCII trailer appended by the scanner software.
    """

    def __init__(self, path,
                 d_bounds=(0, np.inf),
                 e_bounds=(0, np.inf),
                 c_bounds=(0, np.inf),
                 a_bounds=(0, np.inf),
                 x_bounds=(-np.inf, np.inf),
                 y_bounds=(-np.inf, np.inf)):
        self.header  = {}
        self.frames  = None
        self.tracks  = None
        self.trailer = ''

        with open(path, 'rb') as f:
            self._parse_header(f)
            self._parse_data(f, d_bounds, e_bounds, c_bounds,
                             a_bounds, x_bounds, y_bounds)
            self._parse_trailer(f)

    # ── Helper: read a little-endian int32 ────────────────────────────────────
    @staticmethod
    def _ri(f):
        return struct.unpack('<i', f.read(4))[0]

    @staticmethod
    def _rf(f):
        return struct.unpack('<f', f.read(4))[0]

    # ── Header (48 bytes) ─────────────────────────────────────────────────────
    def _parse_header(self, f):
        ri, rf = self._ri, self._rf
        ps = 1e-4 * rf(f)  # pixel size in cm — read after version/dim fields below

        # Re-read in correct order (version_number is field 0)
        f.seek(0)  # rewind
        h = {
            'version_number':     ri(f),
            'num_x_frames':       ri(f),
            'num_y_frames':       ri(f),
            'num_bins':           ri(f),
            'pixel_size':   1e-4 * rf(f),   # cm per pixel
            'pixels_per_bin':     rf(f),
            'border_limit':       ri(f),
            'contrast_limit':     ri(f),
            'eccentricity_limit': ri(f),
            'M':                  ri(f),
            'frame_width':        ri(f),    # in pixels — converted below
            'frame_height':       ri(f),    # in pixels — converted below
        }
        ps = h['pixel_size']
        h['frame_width']  *= ps   # → cm
        h['frame_height'] *= ps   # → cm
        self.header = h

    # ── Frame + track data ────────────────────────────────────────────────────
    def _parse_data(self, f, d_bounds, e_bounds, c_bounds,
                   a_bounds, x_bounds, y_bounds):
        ps = self.header['pixel_size']
        fw = self.header['frame_width']
        fh = self.header['frame_height']
        num_frames = self.header['num_x_frames'] * self.header['num_y_frames']

        frame_rows = []   # collect frame metadata
        track_chunks = [] # collect filtered track arrays (one ndarray per frame)

        for _ in tqdm(range(num_frames), desc='Reading frames', unit='fr'):
            # ── Frame header (28 bytes) ────────────────────────────────────
            number      = struct.unpack('<i', f.read(4))[0]
            x_pos       = 1e-5 * struct.unpack('<i', f.read(4))[0]  # cm
            y_pos       = 1e-5 * struct.unpack('<i', f.read(4))[0]  # cm
            num_tracks  = struct.unpack('<i', f.read(4))[0]
            f.read(12)   # skip 3 unused int32 fields
            focus       = 1e-2 * struct.unpack('<i', f.read(4))[0]  # µm
            xi          = struct.unpack('<i', f.read(4))[0]          # x index
            yi          = struct.unpack('<i', f.read(4))[0]          # y index

            frame_rows.append((number, x_pos, y_pos, num_tracks,
                               focus, xi, yi))

            if num_tracks == 0:
                continue

            # ── Bulk-read all six track arrays at once ─────────────────────
            # Layout in file: d[n], e[n], c[n], a[n], x[n], y[n]
            d_raw = np.frombuffer(f.read(2 * num_tracks), dtype='<i2')  # int16 → µm after scale
            e_raw = np.frombuffer(f.read(num_tracks),     dtype='<i1')  # int8  eccentricity
            c_raw = np.frombuffer(f.read(num_tracks),     dtype='<i1')  # int8  normal contrast
            a_raw = np.frombuffer(f.read(num_tracks),     dtype='<i1')  # int8  average contrast
            x_raw = np.frombuffer(f.read(2 * num_tracks), dtype='<i2')  # int16 pixel position
            y_raw = np.frombuffer(f.read(2 * num_tracks), dtype='<i2')  # int16 pixel position

            # ── Unit conversion (vectorised) ───────────────────────────────
            d_um = 100.0 * d_raw * ps         # diameter in µm
            x_cm = x_pos - 0.5*fw + x_raw*ps  # absolute x in cm
            y_cm = y_pos - 0.5*fh + y_raw*ps  # absolute y in cm

            # ── Vectorised quality + spatial filter ────────────────────────
            mask = (
                (d_um  >= d_bounds[0]) & (d_um  <= d_bounds[1]) &
                (e_raw >= e_bounds[0]) & (e_raw <= e_bounds[1]) &
                (c_raw >= c_bounds[0]) & (c_raw <= c_bounds[1]) &
                (a_raw >= a_bounds[0]) & (a_raw <= a_bounds[1]) &
                (x_cm  >= x_bounds[0]) & (x_cm  <= x_bounds[1]) &
                (y_cm  >= y_bounds[0]) & (y_cm  <= y_bounds[1])
            )

            n_pass = mask.sum()
            if n_pass == 0:
                continue

            # Pack selected tracks into a (n_pass × 7) float64 array
            chunk = np.empty((n_pass, 7), dtype=np.float64)
            chunk[:, 0] = number          # frame_number (stored as float, cast later)
            chunk[:, 1] = d_um[mask]      # d  [µm]
            chunk[:, 2] = x_cm[mask]      # x  [cm]
            chunk[:, 3] = y_cm[mask]      # y  [cm]
            chunk[:, 4] = e_raw[mask]     # e  (eccentricity)
            chunk[:, 5] = c_raw[mask]     # c  (normal contrast)
            chunk[:, 6] = a_raw[mask]     # a  (average contrast)
            track_chunks.append(chunk)

        # ── Build DataFrames once at the end (avoids costly pd.concat in loop) ─
        self.frames = pd.DataFrame(frame_rows, columns=[
            'number', 'x_position', 'y_position', 'num_tracks',
            'focus', 'x_position_index', 'y_position_index'
        ])

        if track_chunks:
            arr = np.vstack(track_chunks)
            self.tracks = pd.DataFrame(arr, columns=[
                'frame_number', 'd', 'x', 'y', 'e', 'c', 'a'
            ])
            # Restore compact dtypes to save memory
            self.tracks['frame_number'] = self.tracks['frame_number'].astype(np.int32)
            self.tracks[['e', 'c', 'a']] = self.tracks[['e', 'c', 'a']].astype(np.int8)
            self.tracks[['d', 'x', 'y']] = self.tracks[['d', 'x', 'y']].astype(np.float32)
        else:
            self.tracks = pd.DataFrame(
                columns=['frame_number', 'd', 'x', 'y', 'e', 'c', 'a'])

    # ── Trailer (ASCII metadata appended by scanner) ──────────────────────────
    def _parse_trailer(self, f):
        f.read(4)  # skip 4-byte separator
        self.trailer = f.read().decode('latin-1')

    def __repr__(self):
        h = self.header
        return (
            f'ScanData  {h["num_x_frames"]}×{h["num_y_frames"]} frames  '
            f'pixel={h["pixel_size"]*1e4:.4f} µm  '
            f'tracks={len(self.tracks):,}'
        )


print('ScanData class defined')