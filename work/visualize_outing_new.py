import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import numpy as np

# --- Configuration ---
OUTINGS_LOG_FILE = 'rule-outing.csv'
DOOR_FAILURE_DAYS_FILE = 'sensors_failure_days/door_failure_days.csv' 
APP_TITLE = "Outings Activity Viewer"
TEXT_COLOR = 'white'
BACKGROUND_COLOR = '#111111'
DATA1_COLOR = '#36A0EB' #bleu
DATA2_COLOR = '#36EB7B' #vert
DATA3_COLOR = '#F14864' #rouge 

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

    # --- Chargement des jours de panne de porte ---
    door_failure_daily_markers = pd.DataFrame(columns=['date'])
    door_failure_source_for_monthly_agg = pd.DataFrame(columns=['failure_count'], index=pd.to_datetime([]))
    door_failure_source_for_monthly_agg.index.name = 'date'
    try:
        failure_dates_df = pd.read_csv(
            DOOR_FAILURE_DAYS_FILE,
            header=None,
            names=['date'],
            parse_dates=[0],
            comment='#'
        )
        door_failure_daily_markers = failure_dates_df[['date']].dropna(subset=['date']).copy()
        door_failure_daily_markers['date'] = pd.to_datetime(door_failure_daily_markers['date'], errors='coerce')
        door_failure_daily_markers.dropna(subset=['date'], inplace=True)

        if not door_failure_daily_markers.empty:
            temp_df_monthly = door_failure_daily_markers.copy()
            temp_df_monthly['failure_count'] = 1
            # Ensure 'date' is datetime index 
            door_failure_source_for_monthly_agg = temp_df_monthly.set_index(pd.to_datetime(temp_df_monthly['date']))
            
    except FileNotFoundError:
        print(f"Avertissement : Fichier '{DOOR_FAILURE_DAYS_FILE}' non trouvé.")
    except pd.errors.EmptyDataError:
        print(f"Avertissement : Fichier '{DOOR_FAILURE_DAYS_FILE}' est vide.")
    except Exception as e:
        print(f"Erreur lors du chargement de '{DOOR_FAILURE_DAYS_FILE}': {e}")

    # --- Daily Aggregation with Duration Spreading ---
    outings_daily_new = pd.DataFrame(columns=['date', 'duration_hours_sum', 'activity_count_sum', 'year_month'])
    outings_daily_new = outings_daily_new.astype({
        'date': 'datetime64[ns]', 'duration_hours_sum': 'float64',
        'activity_count_sum': 'int64', 'year_month': 'object'
    })

    if not activity_raw.empty:
        activity_sorted = activity_raw.sort_index()
        processed_outings = []
        current_outing_start_info = None

        for index_ts, row in activity_sorted.iterrows():
            if row['activity_count'] == 1:
                current_outing_start_info = {'start_ts': index_ts}
            elif row['activity_count'] == 0 and current_outing_start_info is not None:
                start_ts = current_outing_start_info['start_ts']
                end_ts = index_ts
                if pd.notna(row['durationHours']) and row['durationHours'] > 0:
                     processed_outings.append({'start_ts': start_ts, 'end_ts': end_ts})
                current_outing_start_info = None
        
        min_date_overall_activity = activity_sorted.index.min().normalize() if not activity_sorted.index.empty else None
        max_date_overall_activity = activity_sorted.index.max().normalize() if not activity_sorted.index.empty else None
        
        min_date_overall_failure = door_failure_daily_markers['date'].min().normalize() if not door_failure_daily_markers.empty else None
        max_date_overall_failure = door_failure_daily_markers['date'].max().normalize() if not door_failure_daily_markers.empty else None

        all_min_dates = [d for d in [min_date_overall_activity, min_date_overall_failure] if pd.notna(d)]
        all_max_dates = [d for d in [max_date_overall_activity, max_date_overall_failure] if pd.notna(d)]

        min_date_overall = min(all_min_dates) if all_min_dates else pd.Timestamp.now().normalize()
        max_date_overall = max(all_max_dates) if all_max_dates else pd.Timestamp.now().normalize()


        if pd.notna(min_date_overall) and pd.notna(max_date_overall):
            all_days_idx = pd.date_range(start=min_date_overall, end=max_date_overall, freq='D')
            daily_hours_sum = pd.Series(0.0, index=all_days_idx)
            daily_completed_outings_count = pd.Series(0, index=all_days_idx)

            if processed_outings:
                for outing in processed_outings:
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
                            if daily_hours_sum[day_loop_start_ts] > 24.0:
                                daily_hours_sum[day_loop_start_ts] = 24.0
                        current_day_processing_norm += pd.Timedelta(days=1)
                        if current_day_processing_norm > end_actual_ts.normalize() + pd.Timedelta(days=2) and duration_in_day_hours <=0:
                            break
            
            if 'daily_hours_sum' in locals() and not daily_hours_sum.empty :
                outings_daily_new = pd.DataFrame({
                    'date': daily_hours_sum.index,
                    'duration_hours_sum': daily_hours_sum.values,
                    'activity_count_sum': daily_completed_outings_count.reindex(daily_hours_sum.index, fill_value=0).values
                })
                outings_daily_new['year_month'] = outings_daily_new['date'].dt.to_period('M').astype(str)


    # --- Monthly Aggregation ---
    activity_with_duration = activity_raw[(activity_raw['activity_count'] == 0) & activity_raw['durationHours'].notna()].copy()

    if not activity_with_duration.empty:
        outings_monthly_agg = activity_with_duration.resample('ME').agg(
            activity_count_sum=('activity_count', 'size'), 
            duration_hours_mean=('durationHours', 'mean'),
            duration_hours_sem=('durationHours', 'sem')
        )
    else:
        outings_monthly_agg = pd.DataFrame(columns=['activity_count_sum', 'duration_hours_mean', 'duration_hours_sem'],
                                           index=pd.to_datetime([]))
        outings_monthly_agg.index.name = 'date'

    door_failure_monthly_agg = pd.DataFrame(columns=['door_failure_days_sum_monthly'], index=pd.to_datetime([]))
    door_failure_monthly_agg.index.name = 'date'
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
    else: 
        if not door_failure_monthly_agg.empty:
            outings_monthly = door_failure_monthly_agg.copy()
            outings_monthly['door_failure_days_sum_monthly'] = outings_monthly['door_failure_days_sum_monthly'].fillna(0).astype(int)
            for col in ['activity_count_sum', 'duration_hours_mean', 'duration_hours_sem']:
                 if col not in outings_monthly.columns: outings_monthly[col] = np.nan if 'mean' in col or 'sem' in col else 0


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
        if not door_failure_daily_markers.empty:
            door_failure_daily_markers['year_month'] = door_failure_daily_markers['date'].dt.to_period('M').astype(str)
        else:
            if 'year_month' not in door_failure_daily_markers.columns:
                 door_failure_daily_markers['year_month'] = pd.Series(dtype='str')
    else:
        if 'year_month' not in door_failure_daily_markers.columns:
             door_failure_daily_markers['year_month'] = pd.Series(dtype='str')

    return outings_daily_new, outings_monthly, door_failure_daily_markers


# --- Figure Creation Function ---
def create_outings_figure(daily_data, monthly_data, daily_failure_markers, scale, selected_month):
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
            
            if 'door_failure_days_sum_monthly' in df_monthly.columns: # Modifié pour utiliser le nouveau nom de colonne
                fig.add_trace(go.Scatter(x=df_monthly['month_label'], y=df_monthly['door_failure_days_sum_monthly'], name="Jours échec porte / mois", yaxis='y2', mode='lines+markers', line=dict(color=DATA3_COLOR, dash='dot')))
            
            y2_max_val = 5 
            y2_columns_to_check = ['activity_count_sum', 'door_failure_days_sum_monthly'] 
            for col in y2_columns_to_check:
                if col in df_monthly.columns and pd.notna(df_monthly[col].max()):
                    current_max = max(current_max, df_monthly[col].max())
            if current_max > 0:
                y2_max_val = current_max * 1.1

            fig.update_layout(
                title=dict(text="Activité Sorties : Vue Annuelle (Mensuelle)", x=TITLE_X, y=TITLE_Y),
                xaxis=dict(title=dict(text="Mois")),
                yaxis=dict(title=dict(text="Durée moyenne sortie (heures)"), tickfont=dict(color=DATA1_COLOR)),
                yaxis2=dict(title=dict(text="Comptes"), tickfont=dict(color=DATA3_COLOR), showgrid=False, range=[0,y2_max_val]), 
                legend=LEGEND
            )
        else:
            fig.update_layout(title=dict(text="Outings: No yearly data available"))

    elif scale == 'month' and selected_month:
        df_daily_activity = daily_data[daily_data['year_month'] == selected_month] if not daily_data.empty else pd.DataFrame()

        df_daily_failure_filtered = pd.DataFrame()
        if not daily_failure_markers.empty and 'year_month' in daily_failure_markers.columns:
            filtered_failures = daily_failure_markers[daily_failure_markers['year_month'] == selected_month]
            if not filtered_failures.empty:
                df_daily_failure_filtered = filtered_failures
        
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
            fig.update_layout(
                title=dict(text=f"Activité Sorties : Vue Journalière - {selected_month}", x=TITLE_X, y=TITLE_Y),
                xaxis=dict(title=dict(text="Jour"), tickformat='%d', tickmode='array', tickvals=unique_display_dates, ticktext=[d.strftime('%d') for d in unique_display_dates] if unique_display_dates else [], dtick="D1" if not unique_display_dates else None),
                yaxis=dict(title=dict(text="Temps dehors / jour (heures)"), range=[0, 24.5]),
                yaxis2=dict(title=dict(text="Nb. sorties terminées"), range=[0, 3], showgrid=False),
                legend=LEGEND
            )
        else:
            fig.update_layout(title=dict(text=f"Outings: No daily data or door failures for {selected_month}"))
            
    elif scale == 'month' and not selected_month:
        fig.update_layout(title=dict(text="Outings: Please select a month"))

    if not fig.data:
        fig.update_layout(title=dict(text="Aucune donnée à afficher"))
    return fig

# --- Standalone App Execution ---
if __name__ == '__main__':
    outings_daily_data, outings_monthly_data, door_failure_markers_data = get_outings_data()
    
    available_months = []
    temp_months_set = set()
    if not outings_daily_data.empty and 'year_month' in outings_daily_data.columns:
        temp_months_set.update(outings_daily_data['year_month'].dropna().unique())
    if not door_failure_markers_data.empty and 'year_month' in door_failure_markers_data.columns:
        temp_months_set.update(door_failure_markers_data['year_month'].dropna().unique())
    
    if temp_months_set:
        available_months = sorted(list(temp_months_set))

    app = Dash(__name__)
    app.title = APP_TITLE
    app.layout = html.Div(style={'backgroundColor': BACKGROUND_COLOR, 'color': TEXT_COLOR, 'padding': '20px'}, children=[
        html.H2(APP_TITLE, style={'textAlign': 'center', 'marginBottom': '30px'}),
        html.Div([
            html.Label("Select view scale:", style={'marginRight': '10px'}),
            dcc.RadioItems(
                id='scale-selector-outings',
                options=[{'label': 'Year View (Monthly)', 'value': 'year'}, {'label': 'Month View (Daily)', 'value': 'month'}],
                value='year',
                labelStyle={'display': 'inline-block', 'marginRight': '20px'},
                inputStyle={'marginRight': '5px'}
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
        dcc.Graph(id='outings-activity-graph', style={'height': '60vh'})
    ])

    @app.callback(Output('month-dropdown-container-outings', 'style'), Input('scale-selector-outings', 'value'))
    def toggle_month_dropdown_outings(scale):
        return {'display': 'block' if scale == 'month' else 'none', 'textAlign': 'center', 'marginBottom': '20px'}

    @app.callback(Output('outings-activity-graph', 'figure'), Input('scale-selector-outings', 'value'), Input('month-dropdown-outings', 'value'))
    def update_graph_standalone_outings(scale, selected_month):
        return create_outings_figure(outings_daily_data, outings_monthly_data, door_failure_markers_data, scale, selected_month)

    app.run(debug=True)