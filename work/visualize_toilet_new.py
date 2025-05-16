import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import numpy as np
from datetime import datetime as dt_datetime # Pour type hinting ou conversions explicites

# --- Configuration ---
TOILET_LOG_FILE = 'rule-toilet.csv'
TOILET_FAILURE_DAYS_FILE = 'sensors_failure_days/toilet_failure_days.csv'
APP_TITLE = "Toilet Activity Viewer"
TEXT_COLOR = 'white'
BACKGROUND_COLOR = '#111111'
DATA1_COLOR = '#36A0EB' #blue - Monthly mean duration, Daily total duration, Hourly DURATION
DATA2_COLOR = '#43D37B' # green - Monthly total count, Daily total count
DATA3_COLOR = '#EB9636' # Orange - Monthly mean daily count
FAILURE_MARKER_COLOR = '#F14864' # Rouge
DATAMONTH_COLOR = DATA1_COLOR
DATAMONTH2_COLOR = DATA2_COLOR


# --- Graph Configuration ---
LEGEND = dict(orientation="h",yanchor="bottom",y=1.1,xanchor="center",x=0.5)
MARGIN_CHART = dict(l=70, r=70, b=70, t=150, pad=3)
TITLE_X = 0.06
TITLE_Y = 0.92


# --- Data Loading and Processing Function ---
def get_toilet_data():
    try:
        # activity_raw_timestamps contient les événements bruts avec timestamps complets
        activity_raw_timestamps = pd.read_csv(TOILET_LOG_FILE, delimiter=';', decimal=",",
                                        names=["date", "annotation", "activity_count", "duration"],
                                        parse_dates=["date"], index_col="date")
        # 'duration' est en secondes, convertir en minutes pour duration_min
        activity_raw_timestamps['duration_min'] = activity_raw_timestamps['duration'] / 60.0
    except FileNotFoundError:
        print(f"Error: '{TOILET_LOG_FILE}' not found.")
        activity_raw_timestamps = pd.DataFrame(columns=["annotation", "activity_count", "duration","duration_min"],
                                        index=pd.to_datetime([]))
        activity_raw_timestamps.index.name = 'date'
    except Exception as e:
        print(f"Error loading {TOILET_LOG_FILE}: {e}")
        activity_raw_timestamps = pd.DataFrame(columns=["annotation", "activity_count", "duration","duration_min"],
                                        index=pd.to_datetime([]))
        activity_raw_timestamps.index.name = 'date'

    # --- Daily Aggregation ---
    if not activity_raw_timestamps.empty:
        activity_daily_intermediate = activity_raw_timestamps.resample('D').agg(
            activity_count_sum_daily=('activity_count', 'sum'),
            duration_min_sum_daily=('duration_min', 'sum') # Somme des durées des événements qui COMMENCENT ce jour-là
        )
        # activity_daily_for_graph est utilisé pour la vue 'month' et pour peupler les dropdowns de jours
        activity_daily_for_graph = activity_raw_timestamps.resample('D').agg(
            activity_count_sum=('activity_count', 'sum'),
            duration_min_sum=('duration_min', 'sum') # idem
        ).reset_index()
        activity_daily_for_graph['year_month'] = activity_daily_for_graph['date'].dt.strftime('%Y-%m') # Standardisé
        activity_daily_for_graph['date_str'] = activity_daily_for_graph['date'].dt.strftime('%Y-%m-%d') # Ajouté
    else:
        activity_daily_intermediate = pd.DataFrame(
            columns=['activity_count_sum_daily', 'duration_min_sum_daily'],
            index=pd.to_datetime([])
        )
        activity_daily_intermediate.index.name = 'date'
        activity_daily_for_graph = pd.DataFrame(columns=['date', 'activity_count_sum', 'duration_min_sum', 'year_month', 'date_str'])
        activity_daily_for_graph = activity_daily_for_graph.astype({'date': 'datetime64[ns]'})


    # --- Monthly Aggregation ---
    if not activity_daily_intermediate.empty:
        monthly_agg_input = activity_daily_intermediate.copy()
        # Pour la moyenne, on ne veut pas que les jours sans activité (durée 0) tirent la moyenne vers le bas.
        # Si un jour a 0 min, il ne compte pas dans la moyenne des durées journalières.
        monthly_agg_input['duration_min_sum_daily_for_avg'] = monthly_agg_input['duration_min_sum_daily'].replace(0, np.nan)
        activity_monthly = monthly_agg_input.resample('ME').agg(
            duration_min_mean_of_daily_totals=('duration_min_sum_daily_for_avg', 'mean'),
            duration_min_sem_of_daily_totals=('duration_min_sum_daily_for_avg', 'sem'),
            activity_count_sum_monthly=('activity_count_sum_daily', 'sum')
        ).reset_index()
        activity_monthly = activity_monthly.rename(columns={
            'duration_min_mean_of_daily_totals': 'duration_min_mean',
            'duration_min_sem_of_daily_totals': 'duration_min_sem',
            'activity_count_sum_monthly': 'activity_count_sum'
        })
        if pd.api.types.is_datetime64_any_dtype(activity_monthly['date']):
            activity_monthly['days_in_month'] = activity_monthly['date'].dt.daysinmonth
            activity_monthly['activity_count_mean_daily'] = np.where(
                activity_monthly['days_in_month'] > 0,
                activity_monthly['activity_count_sum'] / activity_monthly['days_in_month'],0)
        else:
            activity_monthly['days_in_month'] = 0
            activity_monthly['activity_count_mean_daily'] = 0
    else:
        activity_monthly = pd.DataFrame(columns=['date', 'activity_count_sum', 'duration_min_mean',
                                                'duration_min_sem', 'days_in_month',
                                                'activity_count_mean_daily'])
        if 'date' not in activity_monthly.columns:
            activity_monthly['date'] = pd.Series(dtype='datetime64[ns]')


    # --- Failure Data ---
    toilet_failure_daily_markers = pd.DataFrame(columns=['date', 'year_month'])
    try:
        failure_dates_df = pd.read_csv(
            TOILET_FAILURE_DAYS_FILE, header=None, names=['date'], parse_dates=[0], comment='#'
        )
        toilet_failure_daily_markers_temp = failure_dates_df[['date']].dropna(subset=['date']).copy()
        toilet_failure_daily_markers_temp['date'] = pd.to_datetime(toilet_failure_daily_markers_temp['date'], errors='coerce')
        toilet_failure_daily_markers_temp.dropna(subset=['date'], inplace=True)
        if not toilet_failure_daily_markers_temp.empty:
                toilet_failure_daily_markers_temp['year_month'] = toilet_failure_daily_markers_temp['date'].dt.strftime('%Y-%m')
        toilet_failure_daily_markers = toilet_failure_daily_markers_temp

    except FileNotFoundError:
        print(f"Avertissement : Fichier '{TOILET_FAILURE_DAYS_FILE}' non trouvé.")
    except pd.errors.EmptyDataError:
        print(f"Avertissement : Fichier '{TOILET_FAILURE_DAYS_FILE}' est vide.")
    except Exception as e:
        print(f"Erreur lors du chargement de '{TOILET_FAILURE_DAYS_FILE}': {e}")


    toilet_failure_monthly_sum_agg = pd.DataFrame(columns=['toilet_failure_days_sum'], index=pd.to_datetime([]))
    toilet_failure_monthly_sum_agg.index.name = 'date'
    if not toilet_failure_daily_markers.empty:
        temp_failure_monthly = toilet_failure_daily_markers.copy()
        temp_failure_monthly['date'] = pd.to_datetime(temp_failure_monthly['date'])
        temp_failure_monthly = temp_failure_monthly.set_index('date')
        temp_failure_monthly['failure_day_count'] = 1
        toilet_failure_monthly_sum_agg = temp_failure_monthly.resample('ME').agg(
            toilet_failure_days_sum=('failure_day_count', 'sum')
        )

    if not activity_monthly.empty and 'date' in activity_monthly.columns:
        activity_monthly['date'] = pd.to_datetime(activity_monthly['date'])
        activity_monthly = pd.merge(
            activity_monthly, toilet_failure_monthly_sum_agg.reset_index(), on='date', how='outer'
        )
        for col in ['toilet_failure_days_sum', 'activity_count_sum', 'days_in_month', 'activity_count_mean_daily']:
                if col in activity_monthly.columns:
                    activity_monthly[col] = activity_monthly[col].fillna(0).astype(int if 'sum' in col or 'count' in col or 'days_in_month' in col else float)
                else:
                    activity_monthly[col] = 0
        for col in ['duration_min_mean', 'duration_min_sem']:
            if col in activity_monthly.columns:
                activity_monthly[col] = activity_monthly[col].fillna(np.nan)
            else:
                activity_monthly[col] = np.nan

    elif not toilet_failure_monthly_sum_agg.empty :
        activity_monthly = toilet_failure_monthly_sum_agg.reset_index()
        activity_monthly['toilet_failure_days_sum'] = activity_monthly['toilet_failure_days_sum'].fillna(0).astype(int)
        for col in ['activity_count_sum', 'duration_min_mean', 'duration_min_sem', 'days_in_month', 'activity_count_mean_daily']:
            activity_monthly[col] = np.nan if 'mean' in col or 'sem' in col else 0
    else: # Both are empty
        cols_to_ensure = ['date','activity_count_sum', 'duration_min_mean', 'duration_min_sem', 'days_in_month', 'activity_count_mean_daily', 'toilet_failure_days_sum']
        for col in cols_to_ensure:
            if col not in activity_monthly.columns:
                    activity_monthly[col] = pd.Series(dtype='float64' if 'mean' in col or 'sem' in col else ('datetime64[ns]' if col == 'date' else 'int64'))


    if 'date' in activity_monthly.columns and not activity_monthly.empty:
        valid_dates_monthly = activity_monthly['date'].notna()
        activity_monthly['month_label'] = ''
        if valid_dates_monthly.any():
            activity_monthly.loc[valid_dates_monthly, 'month_label'] = activity_monthly.loc[valid_dates_monthly, 'date'].dt.strftime('%m/%y')
    else:
        if 'month_label' not in activity_monthly.columns: activity_monthly['month_label'] = pd.Series(dtype='str')
        if 'toilet_failure_days_sum' not in activity_monthly.columns: activity_monthly['toilet_failure_days_sum'] = 0
        if 'date' not in activity_monthly.columns: activity_monthly['date'] = pd.Series(dtype='datetime64[ns]')

    return activity_raw_timestamps, activity_daily_for_graph, activity_monthly, toilet_failure_daily_markers


# --- Figure Creation Function ---
def create_toilet_figure(raw_ts_data, aggregated_daily_data, monthly_data, daily_failure_markers,
                        scale, selected_month, selected_day):
    fig = go.Figure()
    fig.update_layout(
        template='plotly_dark', paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR), title=dict(font=dict(color=TEXT_COLOR)),
        legend=dict(font=dict(color=TEXT_COLOR)),
        xaxis=dict(title=dict(font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR), gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(title=dict(font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR), gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis2=dict(title=dict(font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR), gridcolor='rgba(255, 255, 255, 0.1)'),
        margin = MARGIN_CHART,
        hoverlabel=dict(font_size=16, font_color="white", namelength=-1)
    )

    if scale == 'year':
        df_monthly = monthly_data
        if not df_monthly.empty:
            fig.add_trace(go.Bar(x=df_monthly['month_label'], y=df_monthly['duration_min_mean'], name="Durée moyenne (min)", error_y=dict(type='data', array=df_monthly['duration_min_sem']), marker_color=DATA1_COLOR))
            fig.add_trace(go.Scatter(x=df_monthly['month_label'], y=df_monthly['activity_count_sum'], name="Passages totaux", yaxis='y2', mode='lines+markers', line=dict(color=DATA2_COLOR)))
            if 'activity_count_mean_daily' in df_monthly.columns:
                fig.add_trace(go.Scatter(x=df_monthly['month_label'], y=df_monthly['activity_count_mean_daily'], name="Passages moyens/jour", yaxis='y2', mode='lines+markers', line=dict(color=DATA3_COLOR)))
            if 'toilet_failure_days_sum' in df_monthly.columns:
                fig.add_trace(go.Scatter(
                    x=df_monthly['month_label'], y=df_monthly['toilet_failure_days_sum'], name="Jours échec capteur",
                    yaxis='y2', mode='lines+markers', line=dict(color=FAILURE_MARKER_COLOR, dash='dot')
                ))
            
            y2_max_val = 5
            y2_columns_to_check = ['activity_count_sum', 'activity_count_mean_daily', 'toilet_failure_days_sum']
            current_max = 0
            for col in y2_columns_to_check:
                if col in df_monthly.columns and pd.notna(df_monthly[col].max()):
                    current_max = max(current_max, df_monthly[col].max())
            if current_max > 0: y2_max_val = current_max * 1.1

            fig.update_layout(
                title=dict(text="Vue Annuelle : Activité mensuelle Toilettes", x = TITLE_X, y = TITLE_Y),
                xaxis=dict(title=dict(text="Mois")),
                yaxis=dict(title=dict(text="Durée moyenne (min)"), tickfont=dict(color=DATA1_COLOR)),
                yaxis2=dict(title=dict(text="Comptes"), tickfont=dict(color=DATA2_COLOR), overlaying='y', side='right', range=[0, y2_max_val], showgrid=False),
                legend=LEGEND
            )
        else:
            fig.update_layout(title=dict(text="Yearly View: No monthly toilet data available"))

    elif scale == 'month' and selected_month:
        df_daily_activity = aggregated_daily_data[aggregated_daily_data['year_month'] == selected_month] if not aggregated_daily_data.empty else pd.DataFrame()
        df_daily_failure_filtered = pd.DataFrame()
        if not daily_failure_markers.empty and 'year_month' in daily_failure_markers.columns:
            df_daily_failure_filtered = daily_failure_markers[daily_failure_markers['year_month'] == selected_month]
        
        all_relevant_dates = []
        if not df_daily_activity.empty: all_relevant_dates.extend(df_daily_activity['date'].tolist()) 
        if not df_daily_failure_filtered.empty: all_relevant_dates.extend(df_daily_failure_filtered['date'].tolist())
        unique_display_dates = sorted(list(set(all_relevant_dates))) if all_relevant_dates else []

        if not df_daily_activity.empty:
            if 'duration_min_sum' in df_daily_activity.columns:
                fig.add_trace(go.Bar(x=df_daily_activity['date'], y=df_daily_activity['duration_min_sum'], name="Durée totale (min)", yaxis='y1', marker_color = DATAMONTH_COLOR))
            fig.add_trace(go.Scatter(x=df_daily_activity['date'], y=df_daily_activity['activity_count_sum'],name="Passages par jour",yaxis='y2', mode='lines+markers', line=dict(color=DATAMONTH2_COLOR) ))
        if not df_daily_failure_filtered.empty:
            fig.add_trace(go.Scatter(
                x=df_daily_failure_filtered['date'], y=[1.0] * len(df_daily_failure_filtered), name="Échec capteur",
                mode='markers', marker=dict(color=FAILURE_MARKER_COLOR, size=10, symbol='x'), yaxis='y1'
            ))
        
        if not df_daily_activity.empty or not df_daily_failure_filtered.empty:
            y1_max = df_daily_activity['duration_min_sum'].max() if not df_daily_activity.empty and 'duration_min_sum' in df_daily_activity else 0
            y1_range_max = max(40, y1_max * 1.1 if pd.notna(y1_max) else 40)
            y2_max = df_daily_activity['activity_count_sum'].max() if not df_daily_activity.empty else 0
            y2_range_max = max(10, y2_max * 1.1 if pd.notna(y2_max) else 10)

            fig.update_layout(
                title=dict(text=f"Vue Journalière (par jour) Toilettes: {selected_month}", x=TITLE_X, y=TITLE_Y),
                xaxis=dict(title=dict(text="Jour"), tickformat='%d', tickmode='array',
                            tickvals=unique_display_dates,
                            ticktext=[d.strftime('%d') for d in unique_display_dates] if unique_display_dates else []),
                yaxis=dict(title=dict(text="Durée totale (min)"),range=[0, y1_range_max],tickfont=dict(color=DATAMONTH_COLOR)),
                yaxis2=dict(title=dict(text="Nombre de passages"),range=[0, y2_range_max], tickfont=dict(color=DATAMONTH2_COLOR),overlaying='y', side='right',showgrid=False),
                legend=LEGEND
            )
        else:
            fig.update_layout(title=dict(text=f"Monthly View: No toilet data or sensor failures for {selected_month}", x=TITLE_X, y=TITLE_Y))

    elif scale == 'month' and not selected_month:
        fig.update_layout(title=dict(text="Monthly View: Please select a month"))

    elif scale == 'day' and selected_day: # selected_day is 'YYYY-MM-DD' string
        # raw_ts_data index is 'date' (datetime of event start), has 'duration' (seconds), 'duration_min'
        if not raw_ts_data.empty and 'duration' in raw_ts_data.columns:
            try:
                day_start_ts = pd.to_datetime(selected_day + " 00:00:00")
                day_end_ts = day_start_ts + pd.Timedelta(days=1) # Fin de journée (exclusive)

                # Préparer les données avec une colonne 'end_ts' pour chaque visite
                # 'duration' est en secondes dans le fichier original.
                # raw_ts_data.index est le 'start_ts' de la visite.
                visits_on_day_df = raw_ts_data.copy()
                visits_on_day_df['event_start_ts'] = visits_on_day_df.index
                visits_on_day_df['event_end_ts'] = visits_on_day_df['event_start_ts'] + pd.to_timedelta(visits_on_day_df['duration'], unit='s')

                # Filtrer les visites qui chevauchent le jour sélectionné
                relevant_visits = visits_on_day_df[
                    (visits_on_day_df['event_start_ts'] < day_end_ts) &   # La visite commence avant la fin du jour J
                    (visits_on_day_df['event_end_ts'] > day_start_ts)    # La visite se termine après le début du jour J
                ]

                hourly_total_duration_min = pd.Series(0.0, index=range(24))

                if not relevant_visits.empty:
                    for _, visit in relevant_visits.iterrows():
                        visit_actual_start = visit['event_start_ts']
                        visit_actual_end = visit['event_end_ts']

                        # Distribuer la durée de cette visite sur les créneaux horaires du jour sélectionné
                        for hour_of_day in range(24):
                            slot_start_this_hour = day_start_ts + pd.Timedelta(hours=hour_of_day)
                            slot_end_this_hour = slot_start_this_hour + pd.Timedelta(hours=1)

                            # Calculer le chevauchement effectif
                            overlap_start = max(visit_actual_start, slot_start_this_hour)
                            overlap_end = min(visit_actual_end, slot_end_this_hour)

                            if overlap_end > overlap_start: # Il y a un chevauchement
                                duration_in_slot_seconds = (overlap_end - overlap_start).total_seconds()
                                hourly_total_duration_min[hour_of_day] += duration_in_slot_seconds / 60.0
                    
                    # Les valeurs dans hourly_total_duration_min peuvent dépasser 60 si plusieurs visites
                    # ou une longue visite contribuent à la même heure. L'axe Y les coupera à 60.
                    fig.add_trace(go.Bar(
                        x=hourly_total_duration_min.index, # heures 0-23
                        y=hourly_total_duration_min.values,
                        name="Durée aux toilettes (min/heure)",
                        marker_color=DATA1_COLOR 
                    ))
                    fig.update_layout(
                        title=dict(text=f"Vue Horaire : Durée Toilettes le {selected_day}", x=TITLE_X, y=TITLE_Y),
                        xaxis=dict(title="Heure de la journée", tickmode='array',
                                   tickvals=list(range(24)),
                                   ticktext=[f"{h:02d}:00" for h in range(24)]),
                        yaxis=dict(title="Durée aux toilettes (minutes)", 
                                   range=[0, 15]), 
                        yaxis2=dict(overlaying='y', side='right', showticklabels=False, showgrid=False, zeroline=False, title=None), # Cacher yaxis2
                        legend=LEGEND
                    )
                else: # Pas de visites pertinentes pour ce jour
                    fig.update_layout(title=dict(text=f"Day View: No toilet activity for {selected_day}"))
            except Exception as e:
                print(f"Error processing hourly toilet view for {selected_day}: {e}")
                import traceback
                traceback.print_exc()
                fig.update_layout(title=dict(text=f"Error loading toilet data for {selected_day}"))
        else: # Données brutes vides ou colonne 'duration' manquante
            fig.update_layout(title=dict(text="Day View: Raw toilet activity data (or 'duration' column) is not available"))
            
    elif scale == 'day' and not selected_day:
        fig.update_layout(title=dict(text="Day View: Please select a day (after selecting a month)"))


    if not fig.data:
        fig.update_layout(title=dict(text="Aucune donnée à afficher pour la sélection actuelle"))
    return fig

# --- Standalone App Execution ---
if __name__ == '__main__':
    activity_raw_timestamps, activity_daily_data, activity_monthly_data, toilet_failure_markers_data = get_toilet_data()

    available_months = []
    temp_months_set = set()
    if not activity_daily_data.empty and 'year_month' in activity_daily_data.columns:
        temp_months_set.update(activity_daily_data['year_month'].dropna().unique())
    if temp_months_set:
        available_months = sorted(list(temp_months_set))

    app = Dash(__name__)
    app.title = APP_TITLE
    app.layout = html.Div(style={'backgroundColor': BACKGROUND_COLOR, 'color': TEXT_COLOR, 'padding': '20px'}, children=[
        html.H2(APP_TITLE, style={'textAlign': 'center', 'marginBottom': '30px'}),
        html.Div([
            html.Label("Select view scale:", style={'marginRight': '10px'}),
            dcc.Dropdown( 
                id='scale-selector', 
                options=[
                    {'label': 'Year View (Monthly)', 'value': 'year'},
                    {'label': 'Month View (Daily)', 'value': 'month'},
                    {'label': 'Day View (Hourly)', 'value': 'day'} 
                ],
                value='year',
                clearable=False,
                style={'width': '250px', 'display': 'inline-block', 'color': '#333'}
            ),
        ], style={'marginBottom': '20px', 'textAlign': 'center'}),

        html.Div(id='month-dropdown-container', children=[
            html.Label("Select Month:", style={'marginRight': '10px'}),
            dcc.Dropdown(
                id='month-dropdown',
                options=[{'label': m, 'value': m} for m in available_months],
                value=available_months[0] if len(available_months) > 0 else None,
                clearable=False,
                style={'width': '200px', 'display': 'inline-block', 'color': '#333'}
            )
        ], style={'display': 'none', 'textAlign': 'center', 'marginBottom': '20px'}),

        html.Div(id='day-dropdown-container', children=[ 
            html.Label("Select Day:", style={'marginRight': '10px'}),
            dcc.Dropdown(
                id='day-dropdown',
                clearable=False,
                style={'width': '200px', 'display': 'inline-block', 'color': '#333'}
            )
        ], style={'display': 'none', 'textAlign': 'center', 'marginBottom': '20px'}),

        dcc.Graph(id='activity-graph', style={'height': '60vh'})
    ])

    @app.callback(
        Output('month-dropdown-container', 'style'),
        Output('day-dropdown-container', 'style'),
        Input('scale-selector', 'value')
    )
    def toggle_dropdown_visibility(scale): 
        month_style = {'display': 'none', 'textAlign': 'center', 'marginBottom': '20px'}
        day_style = {'display': 'none', 'textAlign': 'center', 'marginBottom': '20px'}
        if scale == 'month':
            month_style['display'] = 'block'
        elif scale == 'day':
            month_style['display'] = 'block'
            day_style['display'] = 'block'
        return month_style, day_style

    @app.callback(
        Output('day-dropdown', 'options'),
        Output('day-dropdown', 'value'),
        Input('month-dropdown', 'value'),
        Input('scale-selector', 'value')
    )
    def update_day_dropdown_options(selected_month, scale): 
        if scale != 'day' or not selected_month:
            return [], None
        options = []
        value = None
        if not activity_daily_data.empty and 'date_str' in activity_daily_data.columns and 'year_month' in activity_daily_data.columns:
            days_in_month_df = activity_daily_data[activity_daily_data['year_month'] == selected_month]
            valid_dates_str = days_in_month_df['date_str'].dropna().unique() 

            available_days_sorted = sorted(list(valid_dates_str))
            options = []
            for d_str in available_days_sorted:
                try:
                    dt_obj = pd.to_datetime(d_str)
                    label = dt_obj.strftime('%d/%m') 
                    options.append({'label': label, 'value': d_str}) 
                except ValueError: 
                    options.append({'label': d_str, 'value': d_str})

            if available_days_sorted:
                value = available_days_sorted[0]
        return options, value

    @app.callback(
        Output('activity-graph', 'figure'),
        Input('scale-selector', 'value'),
        Input('month-dropdown', 'value'),
        Input('day-dropdown', 'value') 
    )
    def update_graph_standalone(scale, selected_month, selected_day): 
        return create_toilet_figure(
            activity_raw_timestamps,    
            activity_daily_data,        
            activity_monthly_data,
            toilet_failure_markers_data,
            scale,
            selected_month,
            selected_day                
        )

    app.run(debug=True)