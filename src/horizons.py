from astroquery.jplhorizons import Horizons
from src.bodies import Body


def fetch_body(name, horizons_id, mass, date, stop_date):
    obj = Horizons(id=horizons_id, location='500@0',
                   epochs={'start': date, 'stop': stop_date, 'step': '1d'})
    vec = obj.vectors()

    x = float(vec['x'][0])
    y = float(vec['y'][0])
    vx = float(vec['vx'][0]) * 365.25   # AU/day -> AU/year
    vy = float(vec['vy'][0]) * 365.25

    return Body(name, mass, [x, y], [vx, vy])
