from __future__ import annotations
import math
from datetime import datetime, time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from lib.db_client import fetch_aggregated

FIELD_TWA="SilverData.WIND_TWA"; FIELD_TWS="SilverData.WIND_TWS"; FIELD_TWD="SilverData.WIND_TWD"
FIELD_PR="SilverData.PERF_BSP_PolarRatio"; FIELD_BSP="SilverData.BSP_BoatSpeed"
FIELD_LAT="SilverData.GPS_Latitude"; FIELD_LON="SilverData.GPS_Longitude"
FIELD_BOBSTAY="LOAD.Bobstay"
FIELD_STBD_BALLAST="ACTUATOR.StbdBallastTankRatio"; FIELD_PORT_BALLAST="ACTUATOR.PortBallastTankRatio"
MAX_DAYS=3; PRE_T0_S=15; DEFAULT_POST_T0_S=30; EDGE_AVG_S=5
PR_START_MIN=70.; PR_END_MIN=50.; BOBSTAY_C60_THRESHOLD=.8
LOOKUP_TOLERANCE_S=1.5; EARTH_R=6371000.; BALLAST_DELTA_MIN=3.; TOP_N_COLORS=15
MODE="gybe"
IS_TACK = MODE == "tack"
PAGE_TITLE = "Détection virements" if IS_TACK else "Détection empannages"


def get_nav_days():
    days=[]
    for x in st.session_state.get("nav_days",[]):
        try:
            d=pd.Timestamp(x).date()
            if d not in days: days.append(d)
        except Exception: pass
    return sorted(days)

def circular_mean_deg(values):
    a=np.asarray(values,float); a=a[np.isfinite(a)]
    if not len(a): return np.nan
    r=np.deg2rad(a); return float(np.rad2deg(np.arctan2(np.mean(np.sin(r)),np.mean(np.cos(r))))%360)

def gps_to_local_xy(lat,lon,lat0,lon0):
    lat=np.asarray(lat,float); lon=np.asarray(lon,float)
    x=EARTH_R*np.deg2rad(lon-float(lon0))*math.cos(math.radians(float(lat0)))
    y=EARTH_R*np.deg2rad(lat-float(lat0)); return x,y

def rotate_ccw(x,y,a_deg):
    a=math.radians(float(a_deg)); return x*math.cos(a)-y*math.sin(a), x*math.sin(a)+y*math.cos(a)

def sign_nonzero(v):
    if not np.isfinite(v) or v==0: return 0
    return 1 if v>0 else -1

def candidate_entry(twa):
    if not np.isfinite(twa): return False
    a=abs(float(twa)); return (30<a<70) if IS_TACK else (120<a<155)

def nearest_row(idx,target):
    if idx.empty:return None
    p=idx.index.get_indexer([target],method="nearest",tolerance=pd.Timedelta(seconds=LOOKUP_TOLERANCE_S))[0]
    return None if p<0 else idx.iloc[p]

@st.cache_data(show_spinner=False,ttl=300)
def fetch_period(start_iso,end_iso):
    fields=[FIELD_TWA,FIELD_TWS,FIELD_TWD,FIELD_PR,FIELD_BSP,FIELD_LAT,FIELD_LON,FIELD_STBD_BALLAST,FIELD_PORT_BALLAST]
    if IS_TACK: fields.append(FIELD_BOBSTAY)
    df=fetch_aggregated(fields_mean=fields,fields_last=[],start_utc_iso=start_iso,end_utc_iso=end_iso,bucket="1s")
    if df is None or df.empty:return pd.DataFrame()
    out=df.copy(); out["time_utc"]=pd.to_datetime(out["time_utc"],utc=True,errors="coerce")
    out=out.dropna(subset=["time_utc"]).sort_values("time_utc").drop_duplicates("time_utc",keep="last").reset_index(drop=True)
    for c in fields:
        if c not in out: out[c]=np.nan
        out[c]=pd.to_numeric(out[c],errors="coerce")
    return out

@st.cache_data(show_spinner=False,ttl=300)
def fetch_day_reference(day_iso):
    d=pd.Timestamp(day_iso).date(); start=pd.Timestamp(datetime.combine(d,time.min),tz="UTC"); end=start+pd.Timedelta(days=1)
    df=fetch_aggregated(fields_mean=[FIELD_PR],fields_last=[],start_utc_iso=start.strftime("%Y-%m-%dT%H:%M:%SZ"),end_utc_iso=end.strftime("%Y-%m-%dT%H:%M:%SZ"),bucket="1m")
    if df is None or df.empty or FIELD_PR not in df:return {"first":None,"last":None}
    df=df.copy(); df["time_utc"]=pd.to_datetime(df["time_utc"],utc=True,errors="coerce"); df[FIELD_PR]=pd.to_numeric(df[FIELD_PR],errors="coerce")
    g=df[df[FIELD_PR]>PR_START_MIN].dropna(subset=["time_utc"])
    return {"first":None,"last":None} if g.empty else {"first":g.time_utc.min(),"last":g.time_utc.max()}

def find_crossings(work):
    """T0 = passage physique réel: 0° pour tack, couture +/-180° pour gybe."""
    twa=work[FIELD_TWA].to_numpy(float); times=pd.DatetimeIndex(work["time_utc"]); out=[]
    for k in range(1,len(work)):
        a,b=twa[k-1],twa[k]
        if not (np.isfinite(a) and np.isfinite(b)): continue
        if IS_TACK:
            if sign_nonzero(a)==sign_nonzero(b) or sign_nonzero(a)==0 or sign_nonzero(b)==0: continue
            # exclude wrap at 180; require both near head-to-wind vicinity
            if abs(a)>90 or abs(b)>90: continue
            denom=abs(a)+abs(b); frac=abs(a)/denom if denom>0 else .5
        else:
            # wrap +180 -> -180 or reverse; both points must be near dead downwind
            if sign_nonzero(a)==sign_nonzero(b) or min(abs(a),abs(b))<170: continue
            da=180-abs(a); db=180-abs(b); denom=da+db; frac=da/denom if denom>0 else .5
        dt=(times[k]-times[k-1]).total_seconds(); out.append(times[k-1]+pd.Timedelta(seconds=frac*dt))
    return out

def detect_ballast_transfer(seq,t0):
    """Detect transfer using >=3 percentage-point reciprocal change, then end at stabilization/extreme."""
    b=seq[[FIELD_STBD_BALLAST,FIELD_PORT_BALLAST]].copy().interpolate(limit=3)
    b=b.dropna()
    if len(b)<5:return None,None
    # baseline from first 3 valid seconds
    base=b.iloc[:min(3,len(b))].median(); ds=b[FIELD_STBD_BALLAST]-base[FIELD_STBD_BALLAST]; dp=b[FIELD_PORT_BALLAST]-base[FIELD_PORT_BALLAST]
    mask=((ds<=-BALLAST_DELTA_MIN)&(dp>=BALLAST_DELTA_MIN))|((ds>=BALLAST_DELTA_MIN)&(dp<=-BALLAST_DELTA_MIN))
    if not mask.any():return None,None
    start=b.index[np.flatnonzero(mask.to_numpy())[0]]
    direction=1 if float(ds.loc[start])>0 else -1  # +: stbd rising, -: port rising
    net=direction*((b[FIELD_STBD_BALLAST]-base[FIELD_STBD_BALLAST])-(b[FIELD_PORT_BALLAST]-base[FIELD_PORT_BALLAST]))
    after=net.loc[start:]
    if after.empty:
        return start,None

    # Fin du transfert :
    # on ne considère PAS la fin de la fenêtre comme une fin de transfert.
    # Il faut avoir atteint le niveau maximal puis disposer d'au moins
    # 3 échantillons valides après ce point pour confirmer que le transfert
    # s'est terminé/stabilisé pendant la trajectoire.
    vals=after.to_numpy(dtype=float)
    imax=int(np.nanargmax(vals))

    # Maximum atteint trop près de la fin => transfert probablement encore
    # en cours à la sortie de la fenêtre : pas de carré de fin.
    if imax >= len(after)-3:
        return start,None

    end=after.index[imax]
    return start,end

def interpolate_trace_value(times,vals,target):
    if target is None or len(times)==0:return None
    x=np.array([(t-times[0]).total_seconds() for t in times],float); y=np.asarray(vals,float); tx=(target-times[0]).total_seconds()
    good=np.isfinite(y)
    if good.sum()<2 or tx<x[good].min() or tx>x[good].max():return None
    return float(np.interp(tx,x[good],y[good]))

def detect(df,day_label,post_t0_s):
    if df.empty:return pd.DataFrame(),{}
    work=df.copy().sort_values("time_utc"); idx=work.set_index("time_utc"); records=[]; traces={}; last_end=None
    for t0 in find_crossings(work):
        t_start=t0-pd.Timedelta(seconds=PRE_T0_S); t_end=t0+pd.Timedelta(seconds=post_t0_s)
        if last_end is not None and t_start<=last_end: continue
        rs=nearest_row(idx,t_start); re=nearest_row(idx,t_end)
        if rs is None or re is None:continue
        twa_entry=rs[FIELD_TWA]; twa_exit=re[FIELD_TWA]; pr_start=rs[FIELD_PR]; pr_end=re[FIELD_PR]
        if not candidate_entry(twa_entry):continue
        if not np.isfinite(pr_start) or pr_start<=PR_START_MIN or not np.isfinite(pr_end) or pr_end<=PR_END_MIN:continue
        if IS_TACK:
            if not np.isfinite(twa_exit) or abs(float(twa_exit)) > 80:
                continue
        else:
            if not np.isfinite(twa_exit) or not (100 < abs(float(twa_exit)) < 170):
                continue
        if sign_nonzero(twa_entry)==sign_nonzero(twa_exit):continue
        seq=idx.loc[t_start:t_end].copy()
        min_pts=max(20,int((PRE_T0_S+post_t0_s)*.65))
        if len(seq)<min_pts:continue
        edges=pd.concat([seq.loc[t_start:t_start+pd.Timedelta(seconds=EDGE_AVG_S-1)],seq.loc[t_end-pd.Timedelta(seconds=EDGE_AVG_S-1):t_end]])
        vw=edges.dropna(subset=[FIELD_TWS,FIELD_TWD])
        if len(vw)<6:continue
        tws_mean=float(vw[FIELD_TWS].mean()); twd_mean=circular_mean_deg(vw[FIELD_TWD])
        gps=seq.dropna(subset=[FIELD_LAT,FIELD_LON]).copy()
        if len(gps)<min_pts:continue
        fg=gps.iloc[0]; x,y=gps_to_local_xy(gps[FIELD_LAT],gps[FIELD_LON],fg[FIELD_LAT],fg[FIELD_LON]); xr,yr=rotate_ccw(x,y,twd_mean)
        mirrored=float(twa_entry)<0
        if mirrored:xr=-xr
        xr=xr-xr[0]; yr=yr-yr[0]
        bsp_i=rs[FIELD_BSP]; bsp_f=re[FIELD_BSP]; bv=seq[FIELD_BSP].dropna(); bsp_min=float(bv.min()) if len(bv) else np.nan
        direction=("Tribord → Bâbord" if twa_entry>0 else "Bâbord → Tribord")
        # symmetrized TWA: requested positive convention for starboard->port, negative original flipped for other direction
        twa_sym=gps[FIELD_TWA].to_numpy(float) * (1.0 if twa_entry>0 else -1.0)
        # yaw-rate proxy from d(TWA)/dt; unwrap prevents false +/-180 spike on gybes
        twa_rad=np.unwrap(np.deg2rad(gps[FIELD_TWA].to_numpy(float))); ts=pd.DatetimeIndex(gps.index); sec=np.array([(t-ts[0]).total_seconds() for t in ts],float)
        good=np.isfinite(twa_rad)&np.isfinite(sec)
        max_yaw=np.nan
        if good.sum()>=3:
            grad=np.gradient(np.rad2deg(twa_rad[good]),sec[good]); max_yaw=float(np.nanmax(np.abs(grad)))
        bstart,bend=detect_ballast_transfer(seq,t0)
        bob=rs.get(FIELD_BOBSTAY,np.nan); c60=bool(IS_TACK and np.isfinite(bob) and float(bob)>BOBSTAY_C60_THRESHOLD)
        mid=f"{day_label} | {t0.strftime('%H:%M:%S')} | {direction} | TWS {tws_mean:.1f}"
        rec={"id":mid,"day":day_label,"t_start_utc":t_start,"t0_utc":t0,"t_end_utc":t_end,"direction":direction,"twa_entry":float(twa_entry),"twa_exit":float(twa_exit),"pr_entry":float(pr_start),"pr_exit":float(pr_end),"tws_mean":tws_mean,"tws_bin":int(math.floor(tws_mean)),"twd_mean":twd_mean,"projected_distance_m":float(yr[-1]),"bsp_initial":float(bsp_i) if np.isfinite(bsp_i) else np.nan,"bsp_min":bsp_min,"bsp_final":float(bsp_f) if np.isfinite(bsp_f) else np.nan,"max_yaw_rate":max_yaw,"ballast_start_utc":bstart,"ballast_end_utc":bend,"c60":c60,"bobstay_entry":float(bob) if np.isfinite(bob) else np.nan}
        records.append(rec)
        rel=np.array([(t-t0).total_seconds() for t in ts],float)
        traces[mid]={"x":np.asarray(xr,float),"y":np.asarray(yr,float),"time":ts,"time_rel_t0_s":rel,"twa_sym":twa_sym,"bsp":gps[FIELD_BSP].to_numpy(float),"projected_distance":np.asarray(yr,float),"lat":gps[FIELD_LAT].to_numpy(float),"lon":gps[FIELD_LON].to_numpy(float),"ballast_start":bstart,"ballast_end":bend}
        last_end=t_end
    return pd.DataFrame(records),traces

def assign_bin_colors(subset):
    """Top 15 projected distances get a ranked continuous palette; remainder grey."""
    ranked=subset.sort_values("projected_distance_m",ascending=False).copy(); top=ranked.head(TOP_N_COLORS)
    cmap=plt.get_cmap("turbo"); colors={}
    n=max(1,len(top))
    for rank,(_,r) in enumerate(top.iterrows()): colors[r["id"]]=cmap(rank/max(1,n-1))
    for mid in ranked.iloc[TOP_N_COLORS:]["id"]: colors[mid]=(0.55,0.55,0.55,0.55)
    return colors

def table_for_bin(subset,selected_ids,post_t0_s):
    t=subset.copy(); t.insert(0,"Afficher",t["id"].isin(selected_ids)); t["T0 UTC"]=t.t0_utc.dt.strftime("%d/%m %H:%M:%S")
    t["Distance axe vent (m)"]=t.projected_distance_m.round(1); t["TWS (nds)"]=t.tws_mean.round(1); t["TWD (°)"]=t.twd_mean.round(1)
    t["BSP init"] = t.bsp_initial.round(1); t["BSP min"]=t.bsp_min.round(1); t["BSP final"]=t.bsp_final.round(1); t["Yaw max (°/s)"]=t.max_yaw_rate.round(1); t["TWA entry (°)"]=t.twa_entry.round(1); t["TWA exit (°)"]=t.twa_exit.round(1)
    t["Ballast début"]=t.ballast_start_utc.apply(lambda x: x.strftime("%H:%M:%S") if pd.notna(x) else "—"); t["Ballast fin"]=t.ballast_end_utc.apply(lambda x: x.strftime("%H:%M:%S") if pd.notna(x) else "—")
    cols=["Afficher","T0 UTC","direction","Distance axe vent (m)","TWS (nds)","TWD (°)","BSP init","BSP min","BSP final","Yaw max (°/s)","TWA entry (°)","TWA exit (°)","Ballast début","Ballast fin"]
    if IS_TACK:
        t["C60"]=t.c60.map({True:"Oui",False:"Non"}); cols.insert(4,"C60")
    return t[cols]

st.set_page_config(page_title=PAGE_TITLE,layout="wide"); st.title(PAGE_TITLE)
st.caption("T0 est le passage physique de la TWA : 0° pour un virement, ±180° pour un empannage. Fenêtre par défaut : T0−15 s à T0+30 s. Heures UTC.")
nav_days=get_nav_days()
if not nav_days: st.warning("Ouvre d'abord la page principale pour charger les journées de navigation."); st.stop()
sel=st.multiselect("Journées à analyser — maximum 3",nav_days,default=[nav_days[-1]],format_func=lambda d:pd.Timestamp(d).strftime("%d/%m/%Y"))
if not sel:st.stop()
if len(sel)>MAX_DAYS:st.error("Maximum 3 journées.");st.stop()
st.subheader("Période analysée")
for d in sorted(sel):
    ref=fetch_day_reference(str(d)); msg=f"{pd.Timestamp(d).strftime('%d/%m/%Y')} — "
    msg += f"BSP_polarRatio > 70 % de {ref['first'].strftime('%H:%M')} à {ref['last'].strftime('%H:%M')} UTC" if ref['first'] is not None else "aucun repère PR > 70 %."
    st.caption(msg)
custom=st.checkbox("Limiter l'analyse avec une heure de début et de fin",False)
if custom:
    a,b=st.columns(2)
    with a: start_time=st.time_input("Heure début UTC",time(0,0),step=60)
    with b: end_time=st.time_input("Heure fin UTC",time(23,59),step=60)
else:start_time=time(0,0);end_time=time(23,59,59)
if custom and end_time<=start_time:st.error("L'heure de fin doit être postérieure à l'heure de début.");st.stop()
post_t0_s=int(st.number_input("Durée après T0 analysée (s)",5,120,DEFAULT_POST_T0_S,1))
all_m=[]; all_tr={}
with st.spinner(f"Lecture DB 1 Hz et {PAGE_TITLE.lower()}..."):
    for d in sorted(sel):
        s=pd.Timestamp(datetime.combine(d,start_time),tz="UTC"); e=pd.Timestamp(datetime.combine(d,end_time),tz="UTC")+pd.Timedelta(seconds=1)
        df=fetch_period(s.strftime("%Y-%m-%dT%H:%M:%SZ"),e.strftime("%Y-%m-%dT%H:%M:%SZ")); m,tr=detect(df,pd.Timestamp(d).strftime("%Y-%m-%d"),post_t0_s)
        if not m.empty:all_m.append(m);all_tr.update(tr)
mdf=pd.concat(all_m,ignore_index=True) if all_m else pd.DataFrame()
if mdf.empty:st.warning("Aucune manœuvre répondant aux critères.");st.stop()
mdf=mdf.sort_values("t0_utc").reset_index(drop=True); st.success(f"{len(mdf)} manœuvre(s) détectée(s).")

def lab(mid):
    r=mdf.loc[mdf.id==mid].iloc[0]; return f"{r.t0_utc.strftime('%d/%m %H:%M:%S')} | {r.direction} | TWS {r.tws_mean:.1f} | {r.projected_distance_m:+.0f}m / {r.bsp_initial:.1f} - {r.bsp_min:.1f} - {r.bsp_final:.1f}"
ids=mdf.id.tolist(); selected_ids=st.multiselect("Manœuvres affichées",ids,default=ids,format_func=lab); shown=mdf[mdf.id.isin(selected_ids)].copy()
if shown.empty:st.info("Aucune manœuvre sélectionnée.");st.stop()

for tws_bin in sorted(shown.tws_bin.unique()):
    sub=shown[shown.tws_bin==tws_bin].copy(); colors=assign_bin_colors(sub)
    st.markdown(f"### TWS {int(tws_bin)}–{int(tws_bin)+1} nds — {len(sub)} manœuvre(s)")
    fig,ax=plt.subplots(figsize=(5.4,5.0))
    for _,r in sub.iterrows():
        tr=all_tr[r.id]; col=colors[r.id]; lw=2.7 if (IS_TACK and r.c60) else 1.45
        ax.plot(tr['x'],tr['y'],color=col,lw=lw,alpha=.9)
        # ballast markers on trajectory
        for bt,mk in [(tr['ballast_start'],'x'),(tr['ballast_end'],'s')]:
            xv=interpolate_trace_value(tr['time'],tr['x'],bt); yv=interpolate_trace_value(tr['time'],tr['y'],bt)
            if xv is not None: ax.scatter([xv],[yv],marker=mk,s=34,color=col,zorder=5)
        # arrival keeps old direction code: green/red circle
        endcol='green' if r.direction=="Tribord → Bâbord" else 'red'; ax.scatter([tr['x'][-1]],[tr['y'][-1]],s=20,color=endcol,zorder=6)
        ax.annotate(f"{r.projected_distance_m:+.0f}m / {r.bsp_initial:.1f} - {r.bsp_min:.1f} - {r.bsp_final:.1f}",(tr['x'][-1],tr['y'][-1]),xytext=(4,3),textcoords='offset points',fontsize=7)
    ax.scatter([0],[0],s=28,color='black',zorder=6); ax.axvline(0,ls='--',lw=.8,alpha=.5);ax.axhline(0,ls=':',lw=.7,alpha=.4);ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel("Distance transversale au vent (m)");ax.set_ylabel("Distance projetée axe du vent (m)");ax.grid(alpha=.2);fig.tight_layout();st.pyplot(fig,use_container_width=False);plt.close(fig)

    fig,ax1=plt.subplots(figsize=(8.4,4.0)); ax2=ax1.twinx();ax3=ax1.twinx();ax3.spines['right'].set_position(('axes',1.12));ax3.patch.set_visible(False)
    for _,r in sub.iterrows():
        tr=all_tr[r.id]; col=colors[r.id];lw=2.5 if (IS_TACK and r.c60) else 1.25;tt=tr['time_rel_t0_s']
        ax1.plot(tt,tr['twa_sym'],color=col,lw=lw,ls='-');ax2.plot(tt,tr['bsp'],color=col,lw=lw,ls='--',alpha=.8);ax3.plot(tt,tr['projected_distance'],color=col,lw=lw,ls=':',alpha=.8)
        # ballast markers ONLY on symmetrized TWA curve
        for bt,mk in [(tr['ballast_start'],'x'),(tr['ballast_end'],'s')]:
            tx=(bt-r.t0_utc).total_seconds() if bt is not None else None; yv=interpolate_trace_value(tr['time'],tr['twa_sym'],bt)
            if tx is not None and yv is not None:ax1.scatter([tx],[yv],marker=mk,s=38,color=col,zorder=6)
    for xx,ls in [(-PRE_T0_S,':'),(0,'--'),(post_t0_s,':')]:ax1.axvline(xx,ls=ls,lw=.9,alpha=.6)
    ax1.set_xlim(-PRE_T0_S,post_t0_s);ax1.set_xlabel("Temps relatif à T0 (s)");ax1.set_ylabel("TWA symétrisé (°)");ax2.set_ylabel("BSP (nds)");ax3.set_ylabel("Distance projetée (m)");ax1.grid(alpha=.2)
    handles=[Line2D([0],[0],color='black',ls='-',label='TWA sym.'),Line2D([0],[0],color='black',ls='--',label='BSP'),Line2D([0],[0],color='black',ls=':',label='Distance'),Line2D([0],[0],marker='x',color='black',ls='None',label='Début transfert ballast'),Line2D([0],[0],marker='s',color='black',ls='None',label='Fin transfert ballast')]
    ax1.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.20),ncol=3,fontsize=8,frameon=False);fig.subplots_adjust(left=.09,right=.78,bottom=.28,top=.96);st.pyplot(fig,use_container_width=False);plt.close(fig)
    st.dataframe(table_for_bin(sub,selected_ids,post_t0_s),use_container_width=True,hide_index=True)

# Map of all filtered trajectory pieces, colored by projected distance
st.subheader("Cartographie des manœuvres filtrées")
map_rows=[]
if not shown.empty:
    vmin=float(shown.projected_distance_m.min()); vmax=float(shown.projected_distance_m.max()); norm=mpl.colors.Normalize(vmin=vmin,vmax=vmax if vmax>vmin else vmin+1); cmap=plt.get_cmap('turbo')
    for _,r in shown.iterrows():
        tr=all_tr[r.id]; rgba=cmap(norm(r.projected_distance_m)); rgb=[int(255*x) for x in rgba[:3]]
        for lat,lon in zip(tr['lat'],tr['lon']):
            if np.isfinite(lat) and np.isfinite(lon):map_rows.append({'lat':lat,'lon':lon,'color':rgb,'distance':r.projected_distance_m,'id':r.id})
if map_rows:
    import pydeck as pdk
    mapdf=pd.DataFrame(map_rows); view=pdk.ViewState(latitude=float(mapdf.lat.mean()),longitude=float(mapdf.lon.mean()),zoom=11,pitch=0)
    layer=pdk.Layer('ScatterplotLayer',data=mapdf,get_position='[lon, lat]',get_fill_color='color',get_radius=2.5,radius_units='meters',pickable=True)
    st.pydeck_chart(pdk.Deck(layers=[layer],initial_view_state=view,tooltip={'text':'Distance axe vent: {distance} m'}),use_container_width=True)
    st.caption("Couleur cartographique = distance projetée sur l’axe du vent (palette commune à toutes les manœuvres affichées).")
else:st.info("Pas de points GPS à cartographier.")
