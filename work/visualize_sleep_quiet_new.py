import pandas as pd
import plotly.graph_objs as go
import numpy as np
from datetime import datetime as dt_datetime
import os

# --- Configuration (now functions can use PARTICIPANT_NUMBER) ---

PARTICIPANT_NUMBER = 1

# TEXT_COLOR, BACKGROUND_COLOR, etc., remain global as they are general styling.
TEXT_COLOR = 'white'
BACKGROUND_COLOR = '#111111'
DATA1_COLOR = '#36A0EB' #bleu
DATA2_COLOR = '#F14864' #rouge
DATA3_COLOR = '#36EB7B' #vert
DATAMONTH_COLOR = '#36A0EB'
SLEEP_HOURLY_COLOR = DATA1_COLOR # Pour 'Dort'
AWAKE_HOURLY_COLOR = DATA3_COLOR # Pour 'Éveil'

# --- Graph Configuration ---
LEGEND = dict(orientation="h",yanchor="bottom",y=1.1,xanchor="center",x=0.5)
MARGIN_CHART = dict(l=70, r=70, b=70, t=150, pad=3)
TITLE_X = 0.06
TITLE_Y = 0.92


# --- Data Loading and Processing Function ---
def get_sleep_data(participant_number):

    PARTICIPANT_NUMBER = participant_number

    """Loads and preprocesses sleep and bed failure data for a given participant."""
    sleep_log_file = f'participant_{PARTICIPANT_NUMBER}/rules/rule-sleep_quiet.csv'
    bed_failure_days_file = f'participant_{PARTICIPANT_NUMBER}/sensors_failure_days/bed_failure_days.csv'
    output_folder = f'participant_{PARTICIPANT_NUMBER}/new_processed_csv/new_sleep_csv'

    try:
        #lecture du csv de sommeil
        sleep_raw_timestamps = pd.read_csv(sleep_log_file, delimiter=';', decimal=",", names=["date", "annotation", "sleep_count", "duration"], parse_dates=["date"], index_col="date")
        #ajout d'une colonne 'durationHr' pour la durée en heures
        sleep_raw_timestamps['durationHr'] = sleep_raw_timestamps['duration'] / 3600.0
    except FileNotFoundError:
        print(f"Error: '{sleep_log_file}' not found.")
        sleep_raw_timestamps = pd.DataFrame(columns=["annotation", "sleep_count", "duration", "durationHr"], index=pd.to_datetime([]))
        sleep_raw_timestamps.index.name = 'date'
    except Exception as e:
        print(f"Error loading {sleep_log_file}: {e}")
        sleep_raw_timestamps = pd.DataFrame(columns=["annotation", "sleep_count", "duration", "durationHr"], index=pd.to_datetime([]))
        sleep_raw_timestamps.index.name = 'date'

    bed_failure_daily_markers = pd.DataFrame(columns=['date'])
    bed_failure_source_for_monthly_agg = pd.DataFrame(columns=['failure_count'], index=pd.to_datetime([]))
    bed_failure_source_for_monthly_agg.index.name = 'date'

    try:
        failure_dates_df = pd.read_csv(
            bed_failure_days_file, header=None, names=['date'], parse_dates=[0], comment='#'
        )
        bed_failure_daily_markers = failure_dates_df[['date']].dropna(subset=['date']).copy()
        bed_failure_daily_markers['date'] = pd.to_datetime(bed_failure_daily_markers['date'], errors='coerce')
        bed_failure_daily_markers.dropna(subset=['date'], inplace=True)

        if not bed_failure_daily_markers.empty:
            temp_bf_monthly = bed_failure_daily_markers.copy()
            temp_bf_monthly['failure_count'] = 1
            bed_failure_source_for_monthly_agg = temp_bf_monthly.set_index('date')
    except FileNotFoundError:
        print(f"Avertissement : Fichier des jours d'échec du lit '{bed_failure_days_file}' non trouvé.")
    except pd.errors.EmptyDataError:
        print(f"Avertissement : Fichier des jours d'échec du lit '{bed_failure_days_file}' est vide.")
    except Exception as e:
        print(f"Erreur lors du chargement du fichier des jours d'échec du lit '{bed_failure_days_file}': {e}")

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

    # -----Monthly Aggregation-----
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


    # --- Create Output Folder and Save CSVs ---
    try:
        os.makedirs(output_folder, exist_ok=True)
        sleep_raw_timestamps.to_csv(os.path.join(output_folder, "processed_sleep_df.csv"), index=False)
        sleep_daily.to_csv(os.path.join(output_folder, "sleep_daily_activity.csv"), index=False)
        sleep_monthly.to_csv(os.path.join(output_folder, "sleep_monthly_activity.csv"), index=False)
        bed_failure_daily_markers.to_csv(os.path.join(output_folder, "bed_sensor_failure_day.csv"), index=False)
        print(f"CSV files saved for participant {PARTICIPANT_NUMBER} in folder: {output_folder}")
    except Exception as e:
        print(f"Error saving CSVs for participant {PARTICIPANT_NUMBER}: {e}")

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
            total_days_in_year = df_monthly['date'].dt.daysinmonth.sum()
            #Met les pannes de capteur en pourcentage
            df_monthly['failure_percentage'] = (df_monthly['bed_failure_sum'] / df_monthly['date'].dt.daysinmonth) * 100

            fig.add_trace(go.Bar(x=df_monthly['month_label'], y=df_monthly['duration_mean'], name="Durée moyenne sommeil (h)", error_y=dict(type='data', array=df_monthly['duration_sem']), marker_color=DATA1_COLOR))
            fig.add_trace(go.Scatter(x=df_monthly['month_label'], y=df_monthly['failure_percentage'], name="Jours échec lit (%)", yaxis='y2', mode='lines+markers', line=dict(color=DATA2_COLOR, dash='dot')))

            y2_max_range = 100
            if pd.notna(df_monthly['failure_percentage'].max()) and df_monthly['failure_percentage'].max() > 0:
                y2_max_range = min(df_monthly['failure_percentage'].max() * 1.2, 100)

            fig.update_layout(
                title=dict(text=f"Participant {PARTICIPANT_NUMBER} - Vue Annuelle : Activité de Sommeil Mensuelle et Échec du Lit", x= TITLE_X, y = TITLE_Y),
                xaxis=dict(title=dict(text="Mois")),
                yaxis=dict(title=dict(text="Durée moyenne sommeil (h)"), tickfont=dict(color=DATA1_COLOR)),
                yaxis2=dict(title=dict(text="Jours échec lit (%)"), overlaying='y', side='right', tickfont=dict(color=DATA2_COLOR), showgrid=False, range=[0, 100]),
                legend=LEGEND,
                hovermode='x unified'
            )
        else:
            fig.update_layout(title=dict(text=f"Participant {PARTICIPANT_NUMBER} - Yearly View: No monthly data available"))

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
                title=dict(text=f"Participant {PARTICIPANT_NUMBER} - Vue Journalière (par jour) : {selected_month}", x= TITLE_X, y = TITLE_Y),
                xaxis=dict(title=dict(text="Jour"), tickformat='%d', tickmode='array',
                                tickvals=unique_display_dates,
                                ticktext=[d.strftime('%d') for d in unique_display_dates] if unique_display_dates else []),
                yaxis=dict(title=dict(text="Durée sommeil (h)"), range=[0, 13]),
                legend=LEGEND,
                yaxis2=dict(overlaying='y', side='right', showticklabels=False, showgrid=False, zeroline=False, title=None)
            )

    elif scale == 'month' and not selected_month:
        fig.update_layout(title=dict(text=f"Participant {PARTICIPANT_NUMBER} - Monthly View: Please select a month"))

    elif scale == 'day' and selected_day: # selected_day est une chaîne 'YYYY-MM-DD'
        if not raw_ts_data.empty and 'duration' in raw_ts_data.columns:
            try:
                day_start_ts = pd.to_datetime(selected_day + " 00:00:00")
                day_end_ts = day_start_ts + pd.Timedelta(days=1)

                sleep_periods_raw = raw_ts_data.copy()
                sleep_periods_raw['event_start_ts'] = sleep_periods_raw.index
                sleep_periods_raw['event_end_ts'] = sleep_periods_raw['event_start_ts'] + pd.to_timedelta(sleep_periods_raw['duration'], unit='s')

                # Filter and normalize sleep periods to be within the selected day (00:00-24:00)
                # Any sleep event that starts before day_start_ts or ends after day_end_ts needs to be trimmed.
                sleep_periods_normalized = []
                for _, row in sleep_periods_raw.iterrows():
                    start_ts = max(row['event_start_ts'], day_start_ts)
                    end_ts = min(row['event_end_ts'], day_end_ts)
                    if start_ts < end_ts: # Ensure the segment has a positive duration within the day
                        sleep_periods_normalized.append({'start': start_ts, 'end': end_ts})

                # Sort periods by start time to process them chronologically
                sleep_periods_normalized.sort(key=lambda x: x['start'])

                # Merge overlapping sleep periods
                merged_sleep_periods = []
                if sleep_periods_normalized:
                    current_merged_period = sleep_periods_normalized[0]
                    for i in range(1, len(sleep_periods_normalized)):
                        next_period = sleep_periods_normalized[i]
                        # If next period overlaps or is contiguous with current_merged_period
                        if next_period['start'] <= current_merged_period['end']:
                            current_merged_period['end'] = max(current_merged_period['end'], next_period['end'])
                        else:
                            merged_sleep_periods.append(current_merged_period)
                            current_merged_period = next_period
                    merged_sleep_periods.append(current_merged_period) # Add the last merged period

                Y_SLEEP = 0.75  # Position for 'Dort'
                Y_AWAKE = 1.25  # Position for 'Éveil'

                # --- Generate all segments (Sleep and Awake) ---
                all_display_segments = []
                current_time_pointer = day_start_ts

                for sleep_segment in merged_sleep_periods:
                    # Add awake segment if there's a gap
                    if current_time_pointer < sleep_segment['start']:
                        all_display_segments.append({
                            'start': current_time_pointer,
                            'end': sleep_segment['start'],
                            'state': 'awake'
                        })
                    
                    # Add sleep segment
                    all_display_segments.append({
                        'start': sleep_segment['start'],
                        'end': sleep_segment['end'],
                        'state': 'sleep'
                    })
                    current_time_pointer = sleep_segment['end']

                # Add final awake segment if the day isn't fully covered
                if current_time_pointer < day_end_ts:
                    all_display_segments.append({
                        'start': current_time_pointer,
                        'end': day_end_ts,
                        'state': 'awake'
                    })

                # Plot all collected segments
                sleep_added_to_legend = False
                awake_added_to_legend = False
                for segment in all_display_segments:
                    x_start_hour = (segment['start'] - day_start_ts).total_seconds() / 3600.0
                    x_end_hour = (segment['end'] - day_start_ts).total_seconds() / 3600.0
                    duration_segment_hr = (segment['end'] - segment['start']).total_seconds() / 3600.0

                    if x_start_hour < x_end_hour: # Only plot if duration is positive
                        if segment['state'] == 'sleep':
                            fig.add_trace(go.Scatter(
                                x=[x_start_hour, x_end_hour],
                                y=[Y_SLEEP, Y_SLEEP],
                                mode='lines',
                                line=dict(width=20, color=SLEEP_HOURLY_COLOR),
                                name="Sommeil",
                                showlegend=not sleep_added_to_legend,
                                hoverinfo='text',
                                text=[
                                    f"Sommeil: {segment['start'].strftime('%H:%M')} - {segment['end'].strftime('%H:%M')}<br>Durée: {duration_segment_hr:.2f}h"
                                ]
                            ))
                            sleep_added_to_legend = True
                        else: # segment['state'] == 'awake'
                            fig.add_trace(go.Scatter(
                                x=[x_start_hour, x_end_hour],
                                y=[Y_AWAKE, Y_AWAKE],
                                mode='lines',
                                line=dict(width=20, color=AWAKE_HOURLY_COLOR),
                                name="Éveil",
                                showlegend=not awake_added_to_legend,
                                hoverinfo='text',
                                text=[
                                    f"Éveil: {segment['start'].strftime('%H:%M')} - {segment['end'].strftime('%H:%M')}<br>Durée: {duration_segment_hr:.2f}h"
                                ]
                            ))
                            awake_added_to_legend = True

                fig.update_layout(
                    title=dict(text=f"Participant {PARTICIPANT_NUMBER} - Vue Journalière (Gantt) : Sommeil et Éveil le {selected_day}", x=TITLE_X, y=TITLE_Y),
                    xaxis=dict(
                        title="Heure de la journée",
                        tickmode='array',
                        tickvals=list(range(0, 25, 2)),
                        ticktext=[f"{h:02d}:00" for h in range(0, 25, 2)],
                        range=[0, 24]
                    ),
                    yaxis=dict(
                        title="État",
                        tickmode='array',
                        tickvals=[Y_SLEEP, Y_AWAKE],
                        ticktext=["Dort", "Éveil"],
                        range=[0.25, 1.75],
                        showgrid=True,
                        zeroline=False
                    ),
                    legend=LEGEND,
                    hovermode='x unified'
                )
            except Exception as e:
                print(f"Error processing hourly sleep view for {selected_day}: {e}")
                import traceback
                traceback.print_exc()
                fig.update_layout(title=dict(text=f"Error loading sleep data for {selected_day}"))
        else:
            fig.update_layout(title=dict(text=f"Participant {PARTICIPANT_NUMBER} - Day View: Raw sleep data (or 'duration' column) is not available"))

    elif scale == 'day' and not selected_day:
        fig.update_layout(title=dict(text=f"Participant {PARTICIPANT_NUMBER} - Day View: Please select a day (after selecting a month)"))

    if not fig.data:
        fig.update_layout(title=dict(text=f"Participant {PARTICIPANT_NUMBER} - Aucune donnée à afficher pour la sélection actuelle"))
    return fig

# Removed the if __name__ == '__main__': block as this file will be imported