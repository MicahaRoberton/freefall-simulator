# Freefall Simulator

Fall toward real satellite imagery and terrain anywhere on Earth. Set what's falling (weight, drag), the starting height, speed and angle, and watch altitude, speed, Mach, terminal velocity and time to impact live.

- Physics: gravity that weakens with altitude, 1976 US Standard Atmosphere air density, quadratic drag with a transonic drag rise.
- Imagery: Esri World Imagery (live, down to ~0.6 m/px near the drop zone), with bundled Copernicus Sentinel-2 imagery for five preset places as an offline fallback, and NASA Blue Marble.
- Terrain: Mapzen/AWS Terrain Tiles.
- Address search: OpenStreetMap Nominatim.

Open `index.html` from any static host (GitHub Pages works out of the box).
