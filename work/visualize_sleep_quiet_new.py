import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import numpy as np
from datetime import datetime as dt_datetime # Pour la conversion de chaîne en objet date

# --- Configuration ---
SLEEP_LOG_FILE = 'rule-sleep_quiet.csv'
BED_FAILURE_DAYS_FILE = 'sensors_failure_days/bed_failure_days.csv'
APP_TITLE = "Sleeping Activity Dashboard"
TEXT_COLOR = 'white'
BACKGROUND_COLOR = '#111111'
DATA1_COLOR = '#36A0EB' #bleu - Pour la durée moyenne mensuelle, durée journalière
DATA2_COLOR = '#F14864' #rouge - Pour les échecs capteur
DATAMONTH_COLOR = '#36A0EB' # Utilisé pour la vue mensuelle/journalière (durée sommeil)
SLEEP_HOURLY_COLOR = '#58D68D' # Vert pour les barres de sommeil horaire (en minutes)

# --- Graph Configuration ---
LEGEND = dict(orientation="h",yanchor="bottom",y=1.1,xanchor="center",x=0.5)
MARGIN_CHART = dict(l=70, r=70, b=70, t=150, pad=3)
TITLE_X = 0.06
TITLE_Y = 0.92


# --- Data Loading and Processing Function ---
def get_sleep_data():
    """Loads and preprocesses sleep and bed failure data."""
    try:
        # Important: 'date' ici est un timestamp complet (date et heure)
        # 'duration' est en secondes
        sleep_raw_timestamps = pd.read_csv(SLEEP_LOG_FILE, delimiter=';', decimal=",",
                                        names=["date", "annotation", "sleep_count", "duration"],
                                        parse_dates=["date"], index_col="date")
        # On conserve la colonne 'duration' (en secondes) pour la vue horaire.
        # La colonne 'durationHr' peut être calculée si besoin pour d'autres vues.
        sleep_raw_timestamps['durationHr'] = sleep_raw_timestamps['duration'] / 3600.0
        # Ne pas filtrer les colonnes ici pour garder 'duration' disponible.
        # L'ancienne ligne: sleep_raw_timestamps = sleep_raw_timestamps[['durationHr']] EST SUPPRIMÉE.
    except FileNotFoundError:
        print(f"Error: '{SLEEP_LOG_FILE}' not found.")
        sleep_raw_timestamps = pd.DataFrame(columns=["annotation", "sleep_count", "duration", "durationHr"], index=pd.to_datetime([]))
        sleep_raw_timestamps.index.name = 'date'
    except Exception as e:
        print(f"Error loading {SLEEP_LOG_FILE}: {e}")
        sleep_raw_timestamps = pd.DataFrame(columns=["annotation", "sleep_count", "duration", "durationHr"], index=pd.to_datetime([]))
        sleep_raw_timestamps.index.name = 'date'

    bed_failure_daily_markers = pd.DataFrame(columns=['date'])
    bed_failure_source_for_monthly_agg = pd.DataFrame(columns=['failure_count'], index=pd.to_datetime([]))
    bed_failure_source_for_monthly_agg.index.name = 'date'

    try:
        failure_dates_df = pd.read_csv(
            BED_FAILURE_DAYS_FILE, header=None, names=['date'], parse_dates=[0], comment='#'
        )
        bed_failure_daily_markers = failure_dates_df[['date']].dropna(subset=['date']).copy()
        bed_failure_daily_markers['date'] = pd.to_datetime(bed_failure_daily_markers['date'], errors='coerce')
        bed_failure_daily_markers.dropna(subset=['date'], inplace=True)

        if not bed_failure_daily_markers.empty:
            temp_bf_monthly = bed_failure_daily_markers.copy()
            temp_bf_monthly['failure_count'] = 1
            bed_failure_source_for_monthly_agg = temp_bf_monthly.set_index('date')
    except FileNotFoundError:
        print(f"Avertissement : Fichier des jours d'échec du lit '{BED_FAILURE_DAYS_FILE}' non trouvé.")
    except pd.errors.EmptyDataError:
        print(f"Avertissement : Fichier des jours d'échec du lit '{BED_FAILURE_DAYS_FILE}' est vide.")
    except Exception as e:
        print(f"Erreur lors du chargement du fichier des jours d'échec du lit '{BED_FAILURE_DAYS_FILE}': {e}")

    # ------Daily Aggregation (utilise durationHr pour la somme journalière en heures)------
    if not sleep_raw_timestamps.empty and 'durationHr' in sleep_raw_timestamps.columns:
        # 'date' sera la colonne datetime (début du jour), 'duration_sum' la somme des heures
        sleep_daily = sleep_raw_timestamps.resample('D')['durationHr'].sum().reset_index(name='duration_sum')
        sleep_daily['date_str'] = sleep_daily['date'].dt.strftime('%Y-%m-%d') 
        sleep_daily['year_month'] = sleep_daily['date'].dt.strftime('%Y-%m') 
    else:
        sleep_daily = pd.DataFrame(columns=['date', 'duration_sum', 'date_str', 'year_month'])
        sleep_daily = sleep_daily.astype({'date': 'datetime64[ns]', 'duration_sum': 'float64',
                                        'date_str': 'object', 'year_month': 'object'})

    # -----Monthly Aggregation (basée sur les totaux journaliers en heures)-----
    if not sleep_daily.empty:
        temp_daily_for_agg = sleep_daily[['date', 'duration_sum']].copy()
        temp_daily_for_agg['duration_sum_for_avg'] = temp_daily_for_agg['duration_sum'].replace(0, np.nan)
        sleep_monthly_agg = temp_daily_for_agg.set_index('date').resample('ME').agg(
            duration_mean=('duration_sum_for_avg', 'mean'),
            duration_sem=('duration_sum_for_avg', 'sem')
        )
    else:
        sleep_monthly_agg = pd.DataFrame(columns=['duration_mean', 'duration_sem'], index=pd.to_datetime([]))
        sleep_monthly_agg.index.name = 'date'

    if not bed_failure_source_for_monthly_agg.empty:
        bed_failure_monthly_agg = bed_failure_source_for_monthly_agg.resample('ME').agg(
            bed_failure_sum=('failure_count', 'sum')
        )
    else:
        bed_failure_monthly_agg = pd.DataFrame(columns=['bed_failure_sum'], index=pd.to_datetime([]))
        bed_failure_monthly_agg.index.name = 'date'

    sleep_monthly = pd.merge(sleep_monthly_agg, bed_failure_monthly_agg, left_index=True, right_index=True, how='outer')
    if not sleep_monthly.empty:
        sleep_monthly.index = pd.to_datetime(sleep_monthly.index)
        start_date, end_date = sleep_monthly.index.min(), sleep_monthly.index.max()
        if pd.notna(start_date) and pd.notna(end_date):
            full_idx = pd.date_range(start=start_date, end=end_date, freq='ME')
            sleep_monthly = sleep_monthly.reindex(full_idx)
        sleep_monthly['duration_mean'] = sleep_monthly['duration_mean'].fillna(np.nan)
        sleep_monthly['duration_sem'] = np.where(sleep_monthly['duration_mean'].isna(), np.nan, sleep_monthly['duration_sem'].fillna(0))
        sleep_monthly['bed_failure_sum'] = sleep_monthly['bed_failure_sum'].fillna(0).astype(int)
    sleep_monthly = sleep_monthly.reset_index().rename(columns={'index': 'date'})

    if 'date' in sleep_monthly.columns and not sleep_monthly.empty:
        valid_dates_monthly = sleep_monthly['date'].notna()
        sleep_monthly['month_label'] = ''
        if valid_dates_monthly.any():
            sleep_monthly.loc[valid_dates_monthly, 'month_label'] = sleep_monthly.loc[valid_dates_monthly, 'date'].dt.strftime('%m/%y')
    else:
        sleep_monthly['month_label'] = pd.Series(dtype='str') # Ensure column exists

    if not bed_failure_daily_markers.empty and 'date' in bed_failure_daily_markers.columns:
        bed_failure_daily_markers['year_month'] = bed_failure_daily_markers['date'].dt.strftime('%Y-%m')
    else:
        if 'year_month' not in bed_failure_daily_markers.columns: # Ensure column exists
            bed_failure_daily_markers['year_month'] = pd.Series(dtype='str')

    return sleep_raw_timestamps, sleep_daily, sleep_monthly, bed_failure_daily_markers

# --- Figure Creation Function ---
def create_sleep_figure(raw_ts_data, aggregated_daily_data, monthly_data, daily_failure_markers,
                        scale, selected_month, selected_day):
    """Creates the Plotly figure for sleep activity based on inputs."""
    fig = go.Figure()
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR),
        title=dict(font=dict(color=TEXT_COLOR)),
        legend=dict(font=dict(color=TEXT_COLOR)),
        xaxis=dict(title=dict(font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR), gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(title=dict(font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR), gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis2=dict(title=dict(font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR), gridcolor='rgba(255, 255, 255, 0.1)'),
        margin=MARGIN_CHART,
        hoverlabel=dict(font_size=16, font_color="white", namelength=-1)
    )

    if scale == 'year':
        df_monthly = monthly_data
        if not df_monthly.empty:
            fig.add_trace(go.Bar(x=df_monthly['month_label'], y=df_monthly['duration_mean'], name="Durée moyenne sommeil (h)", error_y=dict(type='data', array=df_monthly['duration_sem']), marker_color=DATA1_COLOR))
            fig.add_trace(go.Scatter(x=df_monthly['month_label'], y=df_monthly['bed_failure_sum'], name="Jours échec lit", yaxis='y2', mode='lines+markers', line=dict(color=DATA2_COLOR, dash='dot')))
            
            y2_max_range = 5
            if pd.notna(df_monthly['bed_failure_sum'].max()) and df_monthly['bed_failure_sum'].max() > 0:
                y2_max_range = df_monthly['bed_failure_sum'].max() * 1.1

            fig.update_layout(
                title=dict(text="Vue Annuelle : Activité de Sommeil Mensuelle", x= TITLE_X, y = TITLE_Y),
                xaxis=dict(title=dict(text="Mois")),
                yaxis=dict(title=dict(text="Durée moyenne sommeil (h)"), tickfont=dict(color=DATA1_COLOR)),
                yaxis2=dict(title=dict(text="Jours échec lit"), overlaying='y', side='right', tickfont=dict(color=DATA2_COLOR), showgrid=False, range=[0, y2_max_range]),
                legend=LEGEND
            )
        else:
            fig.update_layout(title=dict(text="Yearly View: No monthly data available"))

    elif scale == 'month' and selected_month:
        df_daily_sleep = aggregated_daily_data[aggregated_daily_data['year_month'] == selected_month] if not aggregated_daily_data.empty else pd.DataFrame()
        df_daily_failure_markers_filtered = pd.DataFrame()
        if not daily_failure_markers.empty and 'year_month' in daily_failure_markers.columns :
            df_daily_failure_markers_filtered = daily_failure_markers[daily_failure_markers['year_month'] == selected_month]

        all_relevant_dates = []
        if not df_daily_sleep.empty:
            all_relevant_dates.extend(df_daily_sleep['date'].tolist()) 
        if not df_daily_failure_markers_filtered.empty:
            all_relevant_dates.extend(df_daily_failure_markers_filtered['date'].tolist())
        
        unique_display_dates = []
        if all_relevant_dates:
            unique_display_dates = sorted(list(set(all_relevant_dates)))

        if not df_daily_sleep.empty:
            fig.add_trace(go.Bar(x=df_daily_sleep['date'], y=df_daily_sleep['duration_sum'], name="Durée sommeil (h)", marker_color=DATAMONTH_COLOR))
        
        if not df_daily_failure_markers_filtered.empty:
            fig.add_trace(go.Scatter(
                x=df_daily_failure_markers_filtered['date'], y=[0.3] * len(df_daily_failure_markers_filtered),
                name="Échec lit", mode='markers', marker=dict(color=DATA2_COLOR, size=10, symbol='x')
            ))
        
        if not df_daily_sleep.empty or not df_daily_failure_markers_filtered.empty:
            fig.update_layout(
                title=dict(text=f"Vue Journalière (par jour) : {selected_month}", x= TITLE_X, y = TITLE_Y),
                xaxis=dict(title=dict(text="Jour"), tickformat='%d', tickmode='array',
                            tickvals=unique_display_dates,
                            ticktext=[d.strftime('%d') for d in unique_display_dates] if unique_display_dates else []),
                yaxis=dict(title=dict(text="Durée sommeil (h)"), range=[0, 13]), # Max 13h pour vue journalière
                legend=LEGEND,
                yaxis2=dict(overlaying='y', side='right', showticklabels=False, showgrid=False, zeroline=False, title=None) # Cacher yaxis2
            )
        else:
            fig.update_layout(title=dict(text=f"Monthly View: No sleep data or bed failures for {selected_month}"))
            
    elif scale == 'month' and not selected_month:
        fig.update_layout(title=dict(text="Monthly View: Please select a month"))

    elif scale == 'day' and selected_day: # selected_day est une chaîne 'YYYY-MM-DD'
        # raw_ts_data est sleep_raw_timestamps, son index est 'date' (début de l'événement de sommeil)
        # et il doit contenir 'duration' (en secondes).
        if not raw_ts_data.empty and 'duration' in raw_ts_data.columns:
            try:
                day_start_ts = pd.to_datetime(selected_day + " 00:00:00")
                day_end_ts = day_start_ts + pd.Timedelta(days=1) # Fin de journée (exclusive)

                sleep_periods_df = raw_ts_data.copy()
                sleep_periods_df['event_start_ts'] = sleep_periods_df.index
                sleep_periods_df['event_end_ts'] = sleep_periods_df['event_start_ts'] + pd.to_timedelta(sleep_periods_df['duration'], unit='s')

                relevant_sleep_periods = sleep_periods_df[
                    (sleep_periods_df['event_start_ts'] < day_end_ts) &
                    (sleep_periods_df['event_end_ts'] > day_start_ts)
                ]

                hourly_total_duration_min = pd.Series(0.0, index=range(24))

                if not relevant_sleep_periods.empty:
                    for _, sleep_event in relevant_sleep_periods.iterrows():
                        event_actual_start = sleep_event['event_start_ts']
                        event_actual_end = sleep_event['event_end_ts']

                        for hour_of_day in range(24):
                            slot_start_this_hour = day_start_ts + pd.Timedelta(hours=hour_of_day)
                            slot_end_this_hour = slot_start_this_hour + pd.Timedelta(hours=1)

                            overlap_start = max(event_actual_start, slot_start_this_hour)
                            overlap_end = min(event_actual_end, slot_end_this_hour)

                            if overlap_end > overlap_start:
                                duration_in_slot_seconds = (overlap_end - overlap_start).total_seconds()
                                hourly_total_duration_min[hour_of_day] += duration_in_slot_seconds / 60.0 # Conversion en minutes
                    
                    fig.add_trace(go.Bar(
                        x=hourly_total_duration_min.index, # heures 0-23
                        y=hourly_total_duration_min.values, # en minutes
                        name="Durée sommeil (min/heure)",
                        marker_color=SLEEP_HOURLY_COLOR
                    ))
                    fig.update_layout(
                        title=dict(text=f"Vue Horaire : Sommeil le {selected_day}", x=TITLE_X, y=TITLE_Y),
                        xaxis=dict(title="Heure de la journée",
                                   tickmode='array',
                                   tickvals=list(range(24)),
                                   ticktext=[f"{h:02d}:00" for h in range(24)]),
                        yaxis=dict(title="Durée de sommeil (minutes)", 
                                   range=[0, 60]), # Échelle fixe 0-60 minutes
                        legend=LEGEND,
                        yaxis2=dict(overlaying='y', side='right', showticklabels=False, showgrid=False, zeroline=False, title=None) # Cacher yaxis2
                    )
                else:
                    fig.update_layout(title=dict(text=f"Day View: No sleep data for {selected_day}"))
            except Exception as e:
                print(f"Error processing hourly sleep view for {selected_day}: {e}")
                import traceback
                traceback.print_exc()
                fig.update_layout(title=dict(text=f"Error loading sleep data for {selected_day}"))
        else:
            fig.update_layout(title=dict(text="Day View: Raw sleep data (or 'duration' column) is not available"))
            
    elif scale == 'day' and not selected_day:
        fig.update_layout(title=dict(text="Day View: Please select a day (after selecting a month)"))

    if not fig.data:
        fig.update_layout(title=dict(text="Aucune donnée à afficher pour la sélection actuelle"))
    return fig


# --- Standalone App Execution ---
if __name__ == '__main__':
    sleep_raw_ts_data, sleep_daily_data, sleep_monthly_data, sleep_bed_failure_daily_markers = get_sleep_data()

    available_months = []
    temp_months_set = set()
    if not sleep_daily_data.empty and 'year_month' in sleep_daily_data.columns:
        valid_year_months = sleep_daily_data['year_month'].dropna().unique()
        temp_months_set.update(valid_year_months)
    if temp_months_set:
        available_months = sorted(list(temp_months_set))

    app = Dash(__name__)
    app.title = APP_TITLE
    app.layout = html.Div(style={'backgroundColor': BACKGROUND_COLOR, 'color': TEXT_COLOR, 'padding': '20px'}, children=[
        html.H2(APP_TITLE, style={'textAlign': 'center', 'marginBottom': '30px'}),
        html.Div([
            html.Label("Select view scale:", style={'marginRight': '10px'}),
            dcc.Dropdown( 
                id='scale-selector-sleep',
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

        html.Div(id='month-dropdown-container-sleep', children=[
            html.Label("Select Month:", style={'marginRight': '10px'}),
            dcc.Dropdown(
                id='month-dropdown-sleep',
                options=[{'label': m, 'value': m} for m in available_months],
                value=available_months[0] if len(available_months) > 0 else None,
                clearable=False,
                style={'width': '200px', 'display': 'inline-block', 'color': '#333'}
            )
        ], style={'display': 'none', 'textAlign': 'center', 'marginBottom': '20px'}),

        html.Div(id='day-dropdown-container-sleep', children=[ 
            html.Label("Select Day:", style={'marginRight': '10px'}),
            dcc.Dropdown(
                id='day-dropdown-sleep',
                clearable=False,
                style={'width': '200px', 'display': 'inline-block', 'color': '#333'}
            )
        ], style={'display': 'none', 'textAlign': 'center', 'marginBottom': '20px'}),

        dcc.Graph(id='sleep-activity-graph', style={'height': '60vh'})
    ])

    @app.callback(
        Output('month-dropdown-container-sleep', 'style'),
        Output('day-dropdown-container-sleep', 'style'),
        Input('scale-selector-sleep', 'value')
    )
    def toggle_dropdown_visibility_sleep(scale):
        month_style = {'display': 'none', 'textAlign': 'center', 'marginBottom': '20px'}
        day_style = {'display': 'none', 'textAlign': 'center', 'marginBottom': '20px'}
        if scale == 'month':
            month_style['display'] = 'block'
        elif scale == 'day':
            month_style['display'] = 'block'
            day_style['display'] = 'block'
        return month_style, day_style

    @app.callback(
        Output('day-dropdown-sleep', 'options'),
        Output('day-dropdown-sleep', 'value'),
        Input('month-dropdown-sleep', 'value'),
        Input('scale-selector-sleep', 'value') 
    )
    def update_day_dropdown_options_sleep(selected_month, scale):
        if scale != 'day' or not selected_month:
            return [], None

        options = []
        value = None
        if not sleep_daily_data.empty and 'date_str' in sleep_daily_data.columns and 'year_month' in sleep_daily_data.columns:
            days_in_month_df = sleep_daily_data[sleep_daily_data['year_month'] == selected_month]
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
        Output('sleep-activity-graph', 'figure'),
        Input('scale-selector-sleep', 'value'),
        Input('month-dropdown-sleep', 'value'),
        Input('day-dropdown-sleep', 'value') 
    )
    def update_graph_standalone(scale, selected_month, selected_day): 
        return create_sleep_figure(
            sleep_raw_ts_data,    
            sleep_daily_data,     
            sleep_monthly_data,
            sleep_bed_failure_daily_markers,
            scale,
            selected_month,
            selected_day          
        )

    app.run(debug=True)