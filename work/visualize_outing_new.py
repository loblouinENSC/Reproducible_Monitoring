import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import numpy as np
from datetime import datetime as dt_datetime # For type hinting if needed

# --- Configuration ---
OUTINGS_LOG_FILE = 'rule-outing.csv'
DOOR_FAILURE_DAYS_FILE = 'sensors_failure_days/door_failure_days.csv'
APP_TITLE = "Outings Activity Viewer"
TEXT_COLOR = 'white'
BACKGROUND_COLOR = '#111111'
DATA1_COLOR = '#36A0EB' #bleu - Monthly mean duration, Daily total duration
DATA2_COLOR = '#36EB7B' #vert - Monthly count, Daily count, Hourly outing presence
DATA3_COLOR = '#F14864' #rouge - Failures

#--- Graph Configuration ---
LEGEND = dict(orientation="h",yanchor="bottom",y=1.1,xanchor="center",x=0.5)
MARGIN_CHART = dict(l=70, r=70, b=70, t=150, pad=3)
TITLE_X = 0.06
TITLE_Y = 0.92

# --- Data Loading and Processing Function ---
def get_outings_data():
    try:
        activity_raw = pd.read_csv(OUTINGS_LOG_FILE, delimiter=';', decimal=",",
                                   names=["date", "annotation", "activity_count", "duration"],
                                   parse_dates=["date"], index_col="date")
        activity_raw['durationHours'] = activity_raw['duration'] / 3600.0
    except FileNotFoundError:
        print(f"Error: '{OUTINGS_LOG_FILE}' not found.")
        activity_raw = pd.DataFrame(columns=["annotation", "activity_count", "duration", "durationHours"],
                                    index=pd.to_datetime([]))
        activity_raw.index.name = 'date'
    except Exception as e:
        print(f"Error loading {OUTINGS_LOG_FILE}: {e}")
        activity_raw = pd.DataFrame(columns=["annotation", "activity_count", "duration", "durationHours"],
                                    index=pd.to_datetime([]))
        activity_raw.index.name = 'date'

    door_failure_daily_markers = pd.DataFrame(columns=['date'])
    door_failure_source_for_monthly_agg = pd.DataFrame(columns=['failure_count'], index=pd.to_datetime([]))
    door_failure_source_for_monthly_agg.index.name = 'date'
    try:
        failure_dates_df = pd.read_csv(
            DOOR_FAILURE_DAYS_FILE, header=None, names=['date'], parse_dates=[0], comment='#'
        )
        door_failure_daily_markers = failure_dates_df[['date']].dropna(subset=['date']).copy()
        door_failure_daily_markers['date'] = pd.to_datetime(door_failure_daily_markers['date'], errors='coerce')
        door_failure_daily_markers.dropna(subset=['date'], inplace=True)
        if not door_failure_daily_markers.empty:
            temp_df_monthly = door_failure_daily_markers.copy()
            temp_df_monthly['failure_count'] = 1
            door_failure_source_for_monthly_agg = temp_df_monthly.set_index(pd.to_datetime(temp_df_monthly['date']))
    except FileNotFoundError:
        print(f"Avertissement : Fichier '{DOOR_FAILURE_DAYS_FILE}' non trouvé.")
    except pd.errors.EmptyDataError:
        print(f"Avertissement : Fichier '{DOOR_FAILURE_DAYS_FILE}' est vide.")
    except Exception as e:
        print(f"Erreur lors du chargement de '{DOOR_FAILURE_DAYS_FILE}': {e}")

    # --- Processed Outings (for hourly view) and Daily Aggregation ---
    processed_outings_list = [] # For hourly view
    outings_daily_new = pd.DataFrame(columns=['date', 'duration_hours_sum', 'activity_count_sum', 'year_month', 'date_str'])
    outings_daily_new = outings_daily_new.astype({
        'date': 'datetime64[ns]', 'duration_hours_sum': 'float64',
        'activity_count_sum': 'int64', 'year_month': 'object', 'date_str': 'object'
    })

    if not activity_raw.empty:
        activity_sorted = activity_raw.sort_index()
        current_outing_start_info = None
        for index_ts, row in activity_sorted.iterrows():
            if row['activity_count'] == 1: # Outing starts
                current_outing_start_info = {'start_ts': index_ts}
            elif row['activity_count'] == 0 and current_outing_start_info is not None: # Outing ends
                start_ts = current_outing_start_info['start_ts']
                end_ts = index_ts
                # Use duration from the 'end' event if available, otherwise calculate
                actual_duration_hours = row['durationHours']
                if pd.isna(actual_duration_hours) or actual_duration_hours <= 0:
                     actual_duration_hours = (end_ts - start_ts).total_seconds() / 3600.0

                if actual_duration_hours > 0:
                    processed_outings_list.append({'start_ts': start_ts, 'end_ts': end_ts, 'duration_hours': actual_duration_hours})
                current_outing_start_info = None
        
        # Determine overall date range for daily aggregation
        min_date_overall_activity = activity_sorted.index.min().normalize() if not activity_sorted.index.empty else None
        max_date_overall_activity = activity_sorted.index.max().normalize() if not activity_sorted.index.empty else None
        min_date_overall_failure = door_failure_daily_markers['date'].min().normalize() if not door_failure_daily_markers.empty else None
        max_date_overall_failure = door_failure_daily_markers['date'].max().normalize() if not door_failure_daily_markers.empty else None
        all_min_dates = [d for d in [min_date_overall_activity, min_date_overall_failure] if pd.notna(d)]
        all_max_dates = [d for d in [max_date_overall_activity, max_date_overall_failure] if pd.notna(d)]
        min_date_overall = min(all_min_dates) if all_min_dates else pd.Timestamp.now().normalize()
        max_date_overall = max(all_max_dates) if all_max_dates else pd.Timestamp.now().normalize()

        if pd.notna(min_date_overall) and pd.notna(max_date_overall) and min_date_overall <= max_date_overall:
            all_days_idx = pd.date_range(start=min_date_overall, end=max_date_overall, freq='D')
            daily_hours_sum = pd.Series(0.0, index=all_days_idx)
            daily_completed_outings_count = pd.Series(0, index=all_days_idx)

            if processed_outings_list:
                for outing in processed_outings_list:
                    start_actual_ts = outing['start_ts']
                    end_actual_ts = outing['end_ts']
                    day_of_completion = end_actual_ts.normalize()
                    if day_of_completion in daily_completed_outings_count.index:
                        daily_completed_outings_count[day_of_completion] += 1
                    
                    current_day_processing_norm = start_actual_ts.normalize()
                    while current_day_processing_norm <= end_actual_ts.normalize():
                        day_loop_start_ts = current_day_processing_norm
                        day_loop_end_ts = day_loop_start_ts + pd.Timedelta(days=1)
                        segment_start = max(start_actual_ts, day_loop_start_ts)
                        segment_end = min(end_actual_ts, day_loop_end_ts)
                        duration_in_day_hours = 0.0
                        if segment_end > segment_start:
                            duration_in_day_hours = (segment_end - segment_start).total_seconds() / 3600.0
                        if day_loop_start_ts in daily_hours_sum.index:
                            daily_hours_sum[day_loop_start_ts] += duration_in_day_hours
                            if daily_hours_sum[day_loop_start_ts] > 24.0: # Cap duration at 24h/day
                                daily_hours_sum[day_loop_start_ts] = 24.0
                        current_day_processing_norm += pd.Timedelta(days=1)
                        # Safety break for very long outings or potential infinite loops with tiny durations
                        if current_day_processing_norm > end_actual_ts.normalize() + pd.Timedelta(days=2) and duration_in_day_hours <=0.0001:
                            break
            
            if not daily_hours_sum.empty:
                outings_daily_new = pd.DataFrame({
                    'date': daily_hours_sum.index, # This is datetime
                    'duration_hours_sum': daily_hours_sum.values,
                    'activity_count_sum': daily_completed_outings_count.reindex(daily_hours_sum.index, fill_value=0).values
                })
                outings_daily_new['year_month'] = outings_daily_new['date'].dt.strftime('%Y-%m')
                outings_daily_new['date_str'] = outings_daily_new['date'].dt.strftime('%Y-%m-%d')

    processed_outings_df = pd.DataFrame(processed_outings_list)
    if not processed_outings_df.empty:
        processed_outings_df['start_ts'] = pd.to_datetime(processed_outings_df['start_ts'])
        processed_outings_df['end_ts'] = pd.to_datetime(processed_outings_df['end_ts'])


    # --- Monthly Aggregation --- (Uses activity_raw where activity_count == 0 for completed outings)
    # This logic for monthly might need review if 'processed_outings_df' is more accurate for completed events
    activity_completed_events = activity_raw[(activity_raw['activity_count'] == 0) & activity_raw['durationHours'].notna() & (activity_raw['durationHours'] > 0)].copy()

    if not activity_completed_events.empty:
        outings_monthly_agg = activity_completed_events.resample('ME').agg(
            activity_count_sum=('activity_count', 'size'),
            duration_hours_mean=('durationHours', 'mean'),
            duration_hours_sem=('durationHours', 'sem')
        )
    else:
        outings_monthly_agg = pd.DataFrame(columns=['activity_count_sum', 'duration_hours_mean', 'duration_hours_sem'],
                                           index=pd.to_datetime([]))
        outings_monthly_agg.index.name = 'date'

    door_failure_monthly_agg = pd.DataFrame(columns=['door_failure_days_sum_monthly'], index=pd.to_datetime([]))
    if not door_failure_source_for_monthly_agg.empty and 'failure_count' in door_failure_source_for_monthly_agg.columns:
        door_failure_monthly_agg = door_failure_source_for_monthly_agg.resample('ME').agg(
            door_failure_days_sum_monthly=('failure_count', 'sum')
        )
    
    outings_monthly = pd.merge(outings_monthly_agg, door_failure_monthly_agg, left_index=True, right_index=True, how='outer')
    if not outings_monthly.empty:
        outings_monthly.index = pd.to_datetime(outings_monthly.index)
        start_date_monthly, end_date_monthly = outings_monthly.index.min(), outings_monthly.index.max()
        if pd.notna(start_date_monthly) and pd.notna(end_date_monthly):
            full_idx_monthly = pd.date_range(start=start_date_monthly, end=end_date_monthly, freq='ME')
            outings_monthly = outings_monthly.reindex(full_idx_monthly)
        outings_monthly['activity_count_sum'] = outings_monthly['activity_count_sum'].fillna(0).astype(int)
        outings_monthly['duration_hours_mean'] = outings_monthly['duration_hours_mean'].fillna(np.nan)
        outings_monthly['duration_hours_sem'] = np.where(
            outings_monthly['duration_hours_mean'].isna(), np.nan, outings_monthly['duration_hours_sem'].fillna(0)
        )
        outings_monthly['door_failure_days_sum_monthly'] = outings_monthly['door_failure_days_sum_monthly'].fillna(0).astype(int)
    else: # Handle case where both are empty or only failures exist
        if not door_failure_monthly_agg.empty:
            outings_monthly = door_failure_monthly_agg.copy()
            outings_monthly['door_failure_days_sum_monthly'] = outings_monthly['door_failure_days_sum_monthly'].fillna(0).astype(int)
        for col_default, default_val in {'activity_count_sum':0, 'duration_hours_mean':np.nan, 'duration_hours_sem':np.nan}.items():
            if col_default not in outings_monthly.columns: outings_monthly[col_default] = default_val


    outings_monthly = outings_monthly.reset_index().rename(columns={'index': 'date'})
    if 'date' in outings_monthly.columns and not outings_monthly.empty:
        valid_dates_monthly = outings_monthly['date'].notna()
        outings_monthly['month_label'] = ''
        if valid_dates_monthly.any():
            outings_monthly.loc[valid_dates_monthly, 'month_label'] = outings_monthly.loc[valid_dates_monthly, 'date'].dt.strftime('%m/%y')
    else:
        outings_monthly['month_label'] = ''
        if 'door_failure_days_sum_monthly' not in outings_monthly.columns : outings_monthly['door_failure_days_sum_monthly'] = 0
        if 'date' not in outings_monthly.columns: outings_monthly['date'] = pd.Series(dtype='datetime64[ns]')

    if not door_failure_daily_markers.empty and 'date' in door_failure_daily_markers.columns:
        door_failure_daily_markers['year_month'] = door_failure_daily_markers['date'].dt.strftime('%Y-%m')
    else:
        if 'year_month' not in door_failure_daily_markers.columns:
            door_failure_daily_markers['year_month'] = pd.Series(dtype='str')

    return processed_outings_df, outings_daily_new, outings_monthly, door_failure_daily_markers


# --- Figure Creation Function ---
def create_outings_figure(processed_outings_data, aggregated_daily_data, monthly_data, daily_failure_markers,
                          scale, selected_month, selected_day): # Added selected_day & processed_outings_data
    fig = go.Figure()
    fig.update_layout(
        template='plotly_dark', paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR), title=dict(font=dict(color=TEXT_COLOR)),
        legend=dict(font=dict(color=TEXT_COLOR)),
        xaxis=dict(title=dict(font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR), gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(title=dict(font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR), gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis2=dict(title=dict(font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR), gridcolor='rgba(255, 255, 255, 0.1)', overlaying='y', side='right'),
        margin=MARGIN_CHART,
        hoverlabel=dict(font_size=16, font_color="white", namelength=-1)
    )

    if scale == 'year':
        df_monthly = monthly_data
        if not df_monthly.empty:
            fig.add_trace(go.Bar(x=df_monthly['month_label'], y=df_monthly['duration_hours_mean'], name="Durée moy. sortie (heures)", error_y=dict(type='data', array=df_monthly['duration_hours_sem']), marker_color=DATA1_COLOR))
            fig.add_trace(go.Scatter(x=df_monthly['month_label'], y=df_monthly['activity_count_sum'], name="Nb. sorties / mois", yaxis='y2', mode='lines+markers', line=dict(color=DATA2_COLOR)))
            if 'door_failure_days_sum_monthly' in df_monthly.columns:
                fig.add_trace(go.Scatter(x=df_monthly['month_label'], y=df_monthly['door_failure_days_sum_monthly'], name="Jours échec porte / mois", yaxis='y2', mode='lines+markers', line=dict(color=DATA3_COLOR, dash='dot')))
            
            y2_max_val = 5
            y2_columns_to_check = ['activity_count_sum', 'door_failure_days_sum_monthly']
            current_max = 0
            for col in y2_columns_to_check:
                if col in df_monthly.columns and pd.notna(df_monthly[col].max()):
                    current_max = max(current_max, df_monthly[col].max())
            if current_max > 0: y2_max_val = current_max * 1.1

            fig.update_layout(
                title=dict(text="Activité Sorties : Vue Annuelle (Mensuelle)", x=TITLE_X, y=TITLE_Y),
                xaxis=dict(title=dict(text="Mois")),
                yaxis=dict(title=dict(text="Durée moyenne sortie (heures)"), tickfont=dict(color=DATA1_COLOR)),
                yaxis2=dict(title=dict(text="Comptes"), tickfont=dict(color=DATA2_COLOR), showgrid=False, range=[0,y2_max_val]),
                legend=LEGEND
            )
        else:
            fig.update_layout(title=dict(text="Outings: No yearly data available"))

    elif scale == 'month' and selected_month:
        # aggregated_daily_data is outings_daily_new, has 'year_month', 'date' (datetime), 'duration_hours_sum', 'activity_count_sum'
        df_daily_activity = aggregated_daily_data[aggregated_daily_data['year_month'] == selected_month] if not aggregated_daily_data.empty else pd.DataFrame()
        df_daily_failure_filtered = pd.DataFrame()
        if not daily_failure_markers.empty and 'year_month' in daily_failure_markers.columns:
            df_daily_failure_filtered = daily_failure_markers[daily_failure_markers['year_month'] == selected_month]
        
        all_relevant_dates = []
        if not df_daily_activity.empty: all_relevant_dates.extend(df_daily_activity['date'].tolist())
        if not df_daily_failure_filtered.empty: all_relevant_dates.extend(df_daily_failure_filtered['date'].tolist())
        unique_display_dates = sorted(list(set(all_relevant_dates))) if all_relevant_dates else []

        if not df_daily_activity.empty:
            fig.add_trace(go.Bar(x=df_daily_activity['date'], y=df_daily_activity['duration_hours_sum'], name="Temps dehors / jour (heures)", marker_color=DATA1_COLOR))
            fig.add_trace(go.Scatter(x=df_daily_activity['date'], y=df_daily_activity['activity_count_sum'], name="Nb. sorties terminées / jour", yaxis='y2', mode='lines', line=dict(color=DATA2_COLOR)))
        if not df_daily_failure_filtered.empty:
            fig.add_trace(go.Scatter(x=df_daily_failure_filtered['date'], y=[0.3] * len(df_daily_failure_filtered), name="Échec porte", mode='markers', marker=dict(color=DATA3_COLOR, size=10, symbol='x'), yaxis='y1'))
        
        if not df_daily_activity.empty or not df_daily_failure_filtered.empty:
            y2_max = df_daily_activity['activity_count_sum'].max() if not df_daily_activity.empty else 0
            y2_range_max = max(3, y2_max * 1.1 if pd.notna(y2_max) else 3)

            fig.update_layout(
                title=dict(text=f"Activité Sorties : Vue Journalière (par jour) - {selected_month}", x=TITLE_X, y=TITLE_Y),
                xaxis=dict(title=dict(text="Jour"), tickformat='%d', tickmode='array', tickvals=unique_display_dates, ticktext=[d.strftime('%d') for d in unique_display_dates] if unique_display_dates else []),
                yaxis=dict(title=dict(text="Temps dehors / jour (heures)"), range=[0, 24.5]),
                yaxis2=dict(title=dict(text="Nb. sorties terminées"), range=[0, y2_range_max], showgrid=False),
                legend=LEGEND
            )
        else:
            fig.update_layout(title=dict(text=f"Outings: No daily data or door failures for {selected_month}"))
            
    elif scale == 'month' and not selected_month:
        fig.update_layout(title=dict(text="Outings: Please select a month"))

    elif scale == 'day' and selected_day: # selected_day is 'YYYY-MM-DD' string
        if not processed_outings_data.empty and 'start_ts' in processed_outings_data.columns:
            try:
                day_start_ts = pd.to_datetime(selected_day + " 00:00:00")
                day_end_ts = pd.to_datetime(selected_day + " 23:59:59.999999")
                
                hourly_durations = pd.Series(0.0, index=range(24))

                relevant_outings = processed_outings_data[
                    (processed_outings_data['start_ts'] <= day_end_ts) &
                    (processed_outings_data['end_ts'] >= day_start_ts)
                ]

                if not relevant_outings.empty:
                    for _, outing in relevant_outings.iterrows():
                        outing_start = outing['start_ts']
                        outing_end = outing['end_ts']
                        
                        for hour in range(24):
                            hour_slot_start = day_start_ts + pd.Timedelta(hours=hour)
                            hour_slot_end = hour_slot_start + pd.Timedelta(hours=1)
                            
                            # Calculate overlap
                            overlap_start = max(outing_start, hour_slot_start)
                            overlap_end = min(outing_end, hour_slot_end)
                            
                            if overlap_end > overlap_start:
                                duration_in_hour_seconds = (overlap_end - overlap_start).total_seconds()
                                hourly_durations[hour] += duration_in_hour_seconds / 3600.0
                    
                    # Cap duration at 1 hour for each slot, as we're showing presence
                    hourly_durations = hourly_durations.clip(upper=1.0)

                    fig.add_trace(go.Bar(
                        x=hourly_durations.index, # hours 0-23
                        y=hourly_durations.values,
                        name="Présence dehors (par heure)",
                        marker_color=DATA2_COLOR # Green for outing presence
                    ))
                    fig.update_layout(
                        title=dict(text=f"Vue Horaire : Sorties le {selected_day}", x=TITLE_X, y=TITLE_Y),
                        xaxis=dict(title="Heure de la journée", tickmode='array',
                                   tickvals=list(range(24)),
                                   ticktext=[f"{h:02d}:00" for h in range(24)]),
                        yaxis=dict(title="Présence dehors (fraction de l'heure)", range=[0, 1.1]),
                        legend=LEGEND,
                        yaxis2=dict(showticklabels=False, showgrid=False, zeroline=False, title=None) # Hide yaxis2 if not used
                    )
                else:
                    fig.update_layout(title=dict(text=f"Day View: No outings data for {selected_day}"))
            except Exception as e:
                print(f"Error processing hourly outing view for {selected_day}: {e}")
                fig.update_layout(title=dict(text=f"Error loading hourly data for {selected_day}"))
        else:
            fig.update_layout(title=dict(text="Day View: Processed outings data not available or empty"))
            
    elif scale == 'day' and not selected_day:
        fig.update_layout(title=dict(text="Day View: Please select a day (after selecting a month)"))


    if not fig.data:
        fig.update_layout(title=dict(text="Aucune donnée à afficher pour la sélection actuelle"))
    return fig

# --- Standalone App Execution ---
if __name__ == '__main__':
    # get_outings_data returns 4 DataFrames
    processed_outings_df, outings_daily_data, outings_monthly_data, door_failure_markers_data = get_outings_data()
    
    available_months = []
    temp_months_set = set()
    # Use outings_daily_data (which has 'year_month') to populate month dropdown
    if not outings_daily_data.empty and 'year_month' in outings_daily_data.columns:
        temp_months_set.update(outings_daily_data['year_month'].dropna().unique())
    # if not door_failure_markers_data.empty and 'year_month' in door_failure_markers_data.columns:
    #     temp_months_set.update(door_failure_markers_data['year_month'].dropna().unique())
    if temp_months_set:
        available_months = sorted(list(temp_months_set))

    app = Dash(__name__)
    app.title = APP_TITLE
    app.layout = html.Div(style={'backgroundColor': BACKGROUND_COLOR, 'color': TEXT_COLOR, 'padding': '20px'}, children=[
        html.H2(APP_TITLE, style={'textAlign': 'center', 'marginBottom': '30px'}),
        html.Div([
            html.Label("Select view scale:", style={'marginRight': '10px'}),
            dcc.Dropdown( # Changed to Dropdown
                id='scale-selector-outings',
                options=[
                    {'label': 'Year View (Monthly)', 'value': 'year'},
                    {'label': 'Month View (Daily)', 'value': 'month'},
                    {'label': 'Day View (Hourly)', 'value': 'day'} # New option
                ],
                value='year',
                clearable=False,
                style={'width': '250px', 'display': 'inline-block', 'color': '#333'}
            ),
        ], style={'marginBottom': '20px', 'textAlign': 'center'}),

        html.Div(id='month-dropdown-container-outings', children=[
            html.Label("Select Month:", style={'marginRight': '10px'}),
            dcc.Dropdown(
                id='month-dropdown-outings',
                options=[{'label': m, 'value': m} for m in available_months],
                value=available_months[0] if len(available_months) > 0 else None,
                clearable=False,
                style={'width': '200px', 'display': 'inline-block', 'color': '#333'}
            )
        ], style={'display': 'none', 'textAlign': 'center', 'marginBottom': '20px'}),

        html.Div(id='day-dropdown-container-outings', children=[ # New dropdown for days
            html.Label("Select Day:", style={'marginRight': '10px'}),
            dcc.Dropdown(
                id='day-dropdown-outings',
                clearable=False,
                style={'width': '200px', 'display': 'inline-block', 'color': '#333'}
            )
        ], style={'display': 'none', 'textAlign': 'center', 'marginBottom': '20px'}),

        dcc.Graph(id='outings-activity-graph', style={'height': '60vh'})
    ])

    @app.callback(
        Output('month-dropdown-container-outings', 'style'),
        Output('day-dropdown-container-outings', 'style'),
        Input('scale-selector-outings', 'value')
    )
    def toggle_dropdown_visibility_outings(scale):
        month_style = {'display': 'none', 'textAlign': 'center', 'marginBottom': '20px'}
        day_style = {'display': 'none', 'textAlign': 'center', 'marginBottom': '20px'}
        if scale == 'month':
            month_style['display'] = 'block'
        elif scale == 'day':
            month_style['display'] = 'block'
            day_style['display'] = 'block'
        return month_style, day_style

    @app.callback(
        Output('day-dropdown-outings', 'options'),
        Output('day-dropdown-outings', 'value'),
        Input('month-dropdown-outings', 'value'),
        Input('scale-selector-outings', 'value')
    )
    def update_day_dropdown_options_outings(selected_month, scale):
        if scale != 'day' or not selected_month:
            return [], None
        options = []
        value = None
        # Use outings_daily_data (which has 'year_month' and 'date_str')
        if not outings_daily_data.empty and 'date_str' in outings_daily_data.columns and 'year_month' in outings_daily_data.columns:
            days_in_month_df = outings_daily_data[outings_daily_data['year_month'] == selected_month]
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
        Output('outings-activity-graph', 'figure'),
        Input('scale-selector-outings', 'value'),
        Input('month-dropdown-outings', 'value'),
        Input('day-dropdown-outings', 'value') # New Input
    )
    def update_graph_standalone_outings(scale, selected_month, selected_day): # New parameter
        return create_outings_figure(
            processed_outings_df,       # DataFrame of processed outings with start_ts, end_ts
            outings_daily_data,         # Aggregated daily data
            outings_monthly_data,
            door_failure_markers_data,
            scale,
            selected_month,
            selected_day                # Selected day for hourly view
        )

    app.run(debug=True)