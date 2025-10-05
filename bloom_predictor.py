# We acknowledge that portions of this code was created with assistance from an artificial intelligence (AI) model. However, all ideas and concepts are original to the creators.

import requests
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from tkinter import *
from tkinter import ttk, messagebox
import math
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import pydaymet as daymet
    HAVE_DAYMET = True
except Exception:
    HAVE_DAYMET = False

PHENOLOGY_DATA = {
    "Apple (Full Bloom)": {
        "T_base_C": 5.6, "T_upper_C": 30.0, "GDD_Target_C": 195.0, "Biofix_Month": 3,
        "Rain_Sensitivity": 0.8, "Solar_Sensitivity": 1.1
    },
    "Sweet Cherry (Full Bloom)": {
        "T_base_C": 4.4, "T_upper_C": 30.0, "GDD_Target_C": 250.0, "Biofix_Month": 2,
        "Rain_Sensitivity": 1.2, "Solar_Sensitivity": 1.0
    },
    "Winter Wheat (Heading)": {
        "T_base_C": 4.4, "T_upper_C": 45.0, "GDD_Target_C": 610.0, "Biofix_Month": 1,
        "Rain_Sensitivity": 1.5, "Solar_Sensitivity": 0.9
    }
}

POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

def safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def daylength_hours(lat, doy):
    decl = 23.44 * math.cos(math.radians(360*(172-doy)/365.0))
    lat_rad = math.radians(lat)
    decl_rad = math.radians(decl)
    x = -math.tan(lat_rad) * math.tan(decl_rad)
    if x >= 1.0:
        return 0.0
    if x <= -1.0:
        return 24.0
    hour_angle = math.acos(x)
    return 2.0 * math.degrees(hour_angle) / 15.0

def fetch_power_bulk(lon, lat, start_yyyymmdd, end_yyyymmdd, params=None, timeout=60):
    if params is None:
        params = ["T2M_MIN","T2M_MAX","PRECTOT","ALLSKY_SFC_SW_DWN","RH2M"]
    param_str = ",".join(params)
    url = (
        f"{POWER_BASE_URL}?parameters={param_str}&community=RE"
        f"&longitude={lon}&latitude={lat}&start={start_yyyymmdd}&end={end_yyyymmdd}&format=JSON"
    )
    r = requests.get(url, timeout=timeout)
    data = r.json()
    if "properties" not in data or "parameter" not in data["properties"]:
        raise ValueError(f"NASA POWER API invalid response:\n{json.dumps(data, indent=2)}")
    return data["properties"]["parameter"]

def get_daymet_climatology(crop_name, lon, lat):
    crop = PHENOLOGY_DATA[crop_name]
    T_BASE = crop['T_base_C']
    T_UPPER = crop['T_upper_C']

    if HAVE_DAYMET:
        try:
            ds = daymet.get_bycoords([(lon, lat)], ("1980-01-01", "2010-12-31"), variables=['tmin','tmax','prcp','srad'], to_xarray=True)
            def sel_first(var):
                if 'site' in var.dims:
                    return var.isel(site=0, drop=True)
                if 'id' in var.dims:
                    return var.isel(id=0, drop=True)
                return var
            tmin_raw = sel_first(ds['tmin'])
            tmax_raw = sel_first(ds['tmax'])
            times = pd.to_datetime(tmin_raw['time'].values)
            df = pd.DataFrame({'tmin': tmin_raw.values.astype(float), 'tmax': tmax_raw.values.astype(float)}, index=times)
            df['gdd'] = df.apply(lambda r: max(((min(r['tmax'], T_UPPER)+max(r['tmin'], T_BASE))/2.0)-T_BASE, 0.0), axis=1)
            grouped = df.groupby([df.index.month, df.index.day])['gdd'].mean()
            climatology = {}
            for m in range(1,13):
                for d in range(1,32):
                    try:
                        g = float(grouped.loc[(m,d)])
                    except Exception:
                        g = 0.0
                    dt = date(2001,m,d)
                    climatology[(m,d)] = {'TMIN': T_BASE + g - 2.0, 'TMAX': T_BASE + g + 2.0, 'GDD': g, 'PRECIP': 2.0, 'SWDOWN': 200.0}
            return climatology
        except Exception:
            pass

    climatology = {}
    for doy in range(1,366):
        seasonal = 5.0 + 10.0*math.sin((doy-80)/365.0*2*math.pi)
        tmin = seasonal - 4.0
        tmax = seasonal + 6.0
        gdd = max(((min(tmax, T_UPPER)+max(tmin, T_BASE))/2.0)-T_BASE, 0.0)
        dt = date(2001,1,1) + timedelta(days=doy-1)
        climatology[(dt.month, dt.day)] = {'TMIN': tmin, 'TMAX': tmax, 'GDD': gdd, 'PRECIP': 2.0, 'SWDOWN': 180.0}
    return climatology

def daily_gdd_with_modifiers(tmin, tmax, t_base, t_upper, swdown, prectot, rh2m, lat, doy, crop):
    tmin_adj = max(tmin, t_base)
    tmax_adj = min(tmax, t_upper)
    base_gdd = max(((tmin_adj + tmax_adj)/2.0) - t_base, 0.0)
    sw_ref = 200.0
    k_solar = 0.06 * crop.get('Solar_Sensitivity', 1.0)
    solar_factor = k_solar * ((swdown - sw_ref)/100.0) if swdown is not None else 0.0
    if prectot is None:
        precip_factor = 0.0
    else:
        if prectot < 1.0:
            precip_factor = 0.01 * crop.get('Rain_Sensitivity', 1.0)
        elif prectot <= 5.0:
            precip_factor = 0.02 * crop.get('Rain_Sensitivity', 1.0)
        else:
            precip_factor = -0.03 * crop.get('Rain_Sensitivity', 1.0)
    rh_factor = 0.0
    if rh2m is not None:
        if rh2m < 30:
            rh_factor = -0.02
        elif rh2m <= 70:
            rh_factor = 0.01
        else:
            rh_factor = -0.01
    dl = daylength_hours(lat, doy)
    dl_factor = 0.01 * (dl - 12.0)
    multiplier = 1.0 + solar_factor + precip_factor + rh_factor + dl_factor
    adj_gdd = base_gdd * max(multiplier, 0.0)
    return adj_gdd, base_gdd, solar_factor, precip_factor, rh_factor, dl_factor

def estimate_temp_trend(lon, lat, years_back=5):
    today = date.today()
    years = [today.year - k for k in range(years_back,0,-1)]
    yearly_means = []
    for y in years:
        s = f"{y}0101"; e = f"{y}1231"
        try:
            p = fetch_power_bulk(lon, lat, s, e, params=["T2M_MIN","T2M_MAX"])
            tmin = pd.Series(p.get("T2M_MIN", {})).astype(float) if p.get("T2M_MIN") else pd.Series(dtype=float)
            tmax = pd.Series(p.get("T2M_MAX", {})).astype(float) if p.get("T2M_MAX") else pd.Series(dtype=float)
            if len(tmin)==0 or len(tmax)==0:
                continue
            yearly_means.append(((tmin+tmax)/2.0).mean())
        except Exception:
            continue
    if len(yearly_means) < 2:
        return 0.0
    x = np.arange(len(yearly_means))
    y = np.array(yearly_means)
    slope = np.polyfit(x,y,1)[0]
    return float(slope)

def predict_bloom_trend(crop_name, lon, lat, prediction_year, text_widget=None):
    crop = PHENOLOGY_DATA[crop_name]
    T_BASE = crop['T_base_C']
    T_UPPER = crop['T_upper_C']
    TARGET = crop['GDD_Target_C']
    biofix = date(prediction_year, crop['Biofix_Month'], 1)
    climatology = get_daymet_climatology(crop_name, lon, lat)
    today = date.today()
    try:
        trend = estimate_temp_trend(lon, lat, years_back=5)
    except Exception:
        trend = 0.0

    fetch_start = f"{prediction_year-1}0101"
    fetch_end = today.strftime("%Y%m%d") if prediction_year > today.year else f"{prediction_year}1231"
    try:
        power_cache = fetch_power_bulk(lon, lat, fetch_start, fetch_end, params=["T2M_MIN","T2M_MAX","PRECTOT","ALLSKY_SFC_SW_DWN","RH2M"])
    except Exception:
        power_cache = {}

    def pget(param, ymd):
        if not power_cache:
            return None
        return safe_float(power_cache.get(param, {}).get(ymd), None)

    cum_gdd = 0.0
    records = []
    max_days = 366
    for step in range(max_days):
        curr = biofix + timedelta(days=step)
        if curr.year > prediction_year:
            break
        ymd = curr.strftime("%Y%m%d")
        doy = curr.timetuple().tm_yday
        month_day = (curr.month, curr.day)

        tmin = pget("T2M_MIN", ymd)
        tmax = pget("T2M_MAX", ymd)
        prectot = pget("PRECTOT", ymd)
        swdown = pget("ALLSKY_SFC_SW_DWN", ymd)
        rh2m = pget("RH2M", ymd)

        if tmin is None or tmax is None:
            c = climatology.get(month_day)
            if c:
                tmin = c.get('TMIN', T_BASE + 1.0)
                tmax = c.get('TMAX', T_BASE + 7.0)
                prectot = c.get('PRECIP', 1.5) if prectot is None else prectot
                swdown = c.get('SWDOWN', 180.0) if swdown is None else swdown

        if curr > today and trend != 0.0:
            years_ahead = curr.year - today.year
            warming = trend * years_ahead
            tmin += warming + 0.5 * math.sin(doy * 2*math.pi/365)
            tmax += warming + 0.5 * math.cos(doy * 2*math.pi/365)

        tmin_n = safe_float(tmin, T_BASE + 1.0)
        tmax_n = safe_float(tmax, tmin_n + 6.0)
        prectot_n = safe_float(prectot, 1.0)
        swdown_n = safe_float(swdown, 180.0)
        rh2m_n = safe_float(rh2m, None)

        adj_gdd, base_gdd, solar_f, precip_f, rh_f, dl_f = daily_gdd_with_modifiers(
            tmin_n, tmax_n, T_BASE, T_UPPER, swdown_n, prectot_n, rh2m_n, lat, doy, crop
        )

        cum_gdd += adj_gdd
        records.append({
            'date': curr, 'tmin': tmin_n, 'tmax': tmax_n, 'precip': prectot_n, 'swdown': swdown_n, 'rh2m': rh2m_n,
            'base_gdd': base_gdd, 'adj_gdd': adj_gdd, 'solar_f': solar_f, 'precip_f': precip_f, 'rh_f': rh_f, 'dl_f': dl_f,
            'cum_gdd': cum_gdd
        })

        if text_widget is not None and (step % 20) == 0:
            text_widget.update_idletasks()

        if cum_gdd >= TARGET:
            break

    if not records:
        return "ERROR: no records"

    bloom_date = records[-1]['date']
    header = f"**Predicted Bloom Date: {bloom_date.strftime('%Y-%m-%d')}**\n"
    header += f"Crop: {crop_name} | Location: {lat:.4f}, {lon:.4f}\n"
    header += f"Biofix: {biofix.isoformat()} | Target GDD: {TARGET:.1f} C-days\n"
    header += f"Estimated temp trend (°C/year, last 5 years): {trend:.4f}\n\n"
    header += "Date       Tmin  Tmax  Precip  SWdown  RH   baseGDD  solarF  precipF  rhF  dlF  adjGDD  cumGDD\n"
    header += "-"*110 + "\n"
    lines = []
    for r in records:
        dt = r['date'].strftime("%Y-%m-%d")
        line = (f"{dt}  {r['tmin']:6.2f}  {r['tmax']:6.2f}  {r['precip']:6.2f}  "
                f"{r['swdown']:7.1f}  {r['rh2m'] if r['rh2m'] is not None else 'N/A':>4}  "
                f"{r['base_gdd']:7.3f}  {r['solar_f']:7.3f}  {r['precip_f']:7.3f}  "
                f"{r['rh_f']:5.3f}  {r['dl_f']:5.3f}  {r['adj_gdd']:8.3f}  {r['cum_gdd']:8.3f}")
        lines.append(line)

    footer = "\n\nData sources: NASA POWER (bulk) where available; Daymet climatology as fallback.\n"
    footer += "Model modifiers applied: solar, precipitation, relative humidity (if available), day length, and 5-year temp trend for future simulation.\n"
    return header + "\n".join(lines) + footer

def run_gui():
    root = Tk()
    root.title("Hybrid Bloom Predictor — Trend-based Future Simulation")
    root.geometry("1200x800")
    root.state('zoomed')

    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)

    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text="Single Bloom Prediction")

    main1 = ttk.Frame(tab1, padding=12)
    main1.pack(fill='both', expand=True)

    ttk.Label(main1, text="Crop:").grid(row=0, column=0, sticky=W)
    crop_var1 = StringVar(value=list(PHENOLOGY_DATA.keys())[0])
    ttk.Combobox(main1, textvariable=crop_var1, values=list(PHENOLOGY_DATA.keys()), state='readonly', width=36).grid(row=0, column=1, sticky=W)

    ttk.Label(main1, text="Latitude:").grid(row=1, column=0, sticky=W)
    lat_entry1 = ttk.Entry(main1, width=20); lat_entry1.insert(0,"47.3944"); lat_entry1.grid(row=1, column=1, sticky=W)

    ttk.Label(main1, text="Longitude:").grid(row=2, column=0, sticky=W)
    lon_entry1 = ttk.Entry(main1, width=20); lon_entry1.insert(0,"-120.3204"); lon_entry1.grid(row=2, column=1, sticky=W)

    ttk.Label(main1, text="Prediction Year:").grid(row=3, column=0, sticky=W)
    year_entry1 = ttk.Entry(main1, width=20); year_entry1.insert(0,str(date.today().year)); year_entry1.grid(row=3, column=1, sticky=W)

    text_frame1 = ttk.Frame(main1)
    text_frame1.grid(row=8, column=0, columnspan=2, sticky=(N,S,E,W))
    text_frame1.rowconfigure(0, weight=1); text_frame1.columnconfigure(0, weight=1)
    text1 = Text(text_frame1, wrap='none')
    text1.grid(row=0, column=0, sticky=(N,S,E,W))
    vs1 = ttk.Scrollbar(text_frame1, orient='vertical', command=text1.yview)
    hs1 = ttk.Scrollbar(text_frame1, orient='horizontal', command=text1.xview)
    text1.configure(yscrollcommand=vs1.set, xscrollcommand=hs1.set)
    vs1.grid(row=0, column=1, sticky=(N,S)); hs1.grid(row=1, column=0, sticky=(E,W))

    def do_predict1():
        try:
            lat = float(lat_entry1.get()); lon = float(lon_entry1.get()); year = int(year_entry1.get())
        except Exception:
            messagebox.showwarning("Input Error", "Enter valid lat/lon/year.")
            return
        text1.delete('1.0', END)
        text1.insert(END, "Running prediction — computing... please wait.\n")
        root.update()
        try:
            out = predict_bloom_trend(crop_var1.get(), lon, lat, year, text_widget=text1)
            text1.delete('1.0', END); text1.insert(END, out)
        except Exception as e:
            text1.delete('1.0', END); text1.insert(END, f"ERROR during prediction:\n{e}")

    ttk.Button(main1, text="Predict Bloom Date", command=do_predict1).grid(row=5, column=0, pady=6, sticky=W)

    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text="Bloom Trend Graph")

    main2 = ttk.Frame(tab2, padding=12)
    main2.pack(fill='both', expand=True)

    ttk.Label(main2, text="Crop:").grid(row=0, column=0, sticky=W)
    crop_var2 = StringVar(value=list(PHENOLOGY_DATA.keys())[0])
    ttk.Combobox(main2, textvariable=crop_var2, values=list(PHENOLOGY_DATA.keys()), state='readonly', width=36).grid(row=0, column=1, sticky=W)

    ttk.Label(main2, text="Latitude:").grid(row=1, column=0, sticky=W)
    lat_entry2 = ttk.Entry(main2, width=20); lat_entry2.insert(0,"47.3944"); lat_entry2.grid(row=1, column=1, sticky=W)

    ttk.Label(main2, text="Longitude:").grid(row=2, column=0, sticky=W)
    lon_entry2 = ttk.Entry(main2, width=20); lon_entry2.insert(0,"-120.3204"); lon_entry2.grid(row=2, column=1, sticky=W)

    ttk.Label(main2, text="Start Year:").grid(row=3, column=0, sticky=W)
    start_entry2 = ttk.Entry(main2, width=20); start_entry2.insert(0,"2015"); start_entry2.grid(row=3, column=1, sticky=W)

    ttk.Label(main2, text="End Year:").grid(row=4, column=0, sticky=W)
    end_entry2 = ttk.Entry(main2, width=20); end_entry2.insert(0,str(date.today().year)); end_entry2.grid(row=4, column=1, sticky=W)

    fig, ax = plt.subplots(figsize=(10,5))
    canvas = FigureCanvasTkAgg(fig, master=main2)
    canvas.get_tk_widget().grid(row=6, column=0, columnspan=2, sticky=(N,S,E,W))

    def do_graph():
        try:
            lat = float(lat_entry2.get()); lon = float(lon_entry2.get())
            start_year = int(start_entry2.get()); end_year = int(end_entry2.get())
        except Exception:
            messagebox.showwarning("Input Error", "Enter valid lat/lon/start/end years.")
            return
        bloom_dates = []
        years = list(range(start_year, end_year+1))
        text1.delete('1.0', END)
        for y in years:
            try:
                out = predict_bloom_trend(crop_var2.get(), lon, lat, y)
                lines = out.splitlines()
                for line in lines:
                    if line.startswith("**Predicted Bloom Date:"):
                        dt_str = line.split(":")[1].strip(" *")
                        bloom_dates.append(datetime.strptime(dt_str,"%Y-%m-%d").date())
            except Exception:
                bloom_dates.append(None)
        ax.clear()
        ax.plot(years, bloom_dates, marker='o')
        ax.set_title(f"Predicted Bloom Dates: {crop_var2.get()}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Bloom Date")
        ax.grid(True)
        canvas.draw()

    ttk.Button(main2, text="Generate Bloom Trend Graph", command=do_graph).grid(row=5, column=0, pady=6, sticky=W)

    root.mainloop()

if __name__ == "__main__":
    run_gui()
