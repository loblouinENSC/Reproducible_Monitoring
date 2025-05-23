import pandas as pd
import plotly.graph_objs as go
# La partie Dash n'est pas nécessaire ici si ce fichier est importé par dashBoardManager.py
# from dash import Dash, dcc, html, Input, Output
import numpy as np
from datetime import datetime as dt_datetime
import os
import sys # Import sys to access command line arguments

PARTICIPANT_NUMBER = 1 

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
# La fonction get_outings_data prend maintenant PARTICIPANT_NUMBER en paramètre
def get_outings_data(participant_number):

    PARTICIPANT_NUMBER = participant_number

    # Définition des chemins de fichiers en utilisant le paramètre PARTICIPANT_NUMBER
    outings_log_file = f'participant_{PARTICIPANT_NUMBER}/rules/rule-outing.csv'
    door_failure_days_file = f'participant_{PARTICIPANT_NUMBER}/sensors_failure_days/door_failure_days.csv'
    output_folder = f'participant_{PARTICIPANT_NUMBER}/new_processed_csv/new_outing_csv'

    try:
        activity_raw = pd.read_csv(outings_log_file, delimiter=';', decimal=",",
                                     names=["date", "annotation", "activity_count", "duration"],
                                     parse_dates=["date"], index_col="date")
        activity_raw['durationHours'] = activity_raw['duration'] / 3600.0
    except FileNotFoundError:
        print(f"Error: '{outings_log_file}' not found for participant {PARTICIPANT_NUMBER}.")
        activity_raw = pd.DataFrame(columns=["annotation", "activity_count", "duration", "durationHours"],
                                     index=pd.to_datetime([]))
        activity_raw.index.name = 'date'
    except Exception as e:
        print(f"Error loading {outings_log_file} for participant {PARTICIPANT_NUMBER}: {e}")
        activity_raw = pd.DataFrame(columns=["annotation", "activity_count", "duration", "durationHours"],
                                     index=pd.to_datetime([]))
        activity_raw.index.name = 'date'

    door_failure_daily_markers = pd.DataFrame(columns=['date'])
    door_failure_source_for_monthly_agg = pd.DataFrame(columns=['failure_count'], index=pd.to_datetime([]))
    door_failure_source_for_monthly_agg.index.name = 'date'
    try:
        failure_dates_df = pd.read_csv(
            door_failure_days_file, header=None, names=['date'], parse_dates=[0], comment='#'
        )
        door_failure_daily_markers = failure_dates_df[['date']].dropna(subset=['date']).copy()
        door_failure_daily_markers['date'] = pd.to_datetime(door_failure_daily_markers['date'], errors='coerce')
        door_failure_daily_markers.dropna(subset=['date'], inplace=True)
        if not door_failure_daily_markers.empty:
            temp_df_monthly = door_failure_daily_markers.copy()
            temp_df_monthly['failure_count'] = 1
            door_failure_source_for_monthly_agg = temp_df_monthly.set_index(pd.to_datetime(temp_df_monthly['date']))
    except FileNotFoundError:
        print(f"Avertissement : Fichier '{door_failure_days_file}' non trouvé pour participant {PARTICIPANT_NUMBER}.")
    except pd.errors.EmptyDataError:
        print(f"Avertissement : Fichier '{door_failure_days_file}' est vide pour participant {PARTICIPANT_NUMBER}.")
    except Exception as e:
        print(f"Erreur lors du chargement de '{door_failure_days_file}' pour participant {PARTICIPANT_NUMBER}: {e}")

    # --- Processed Outings (for hourly view) and Daily Aggregation ---
    processed_outings_list = []
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
                    'date': daily_hours_sum.index,
                    'duration_hours_sum': daily_hours_sum.values,
                    'activity_count_sum': daily_completed_outings_count.reindex(daily_hours_sum.index, fill_value=0).values
                })
                outings_daily_new['year_month'] = outings_daily_new['date'].dt.strftime('%Y-%m')
                outings_daily_new['date_str'] = outings_daily_new['date'].dt.strftime('%Y-%m-%d')

    processed_outings_df = pd.DataFrame(processed_outings_list)
    if not processed_outings_df.empty:
        processed_outings_df['start_ts'] = pd.to_datetime(processed_outings_df['start_ts'])
        processed_outings_df['end_ts'] = pd.to_datetime(processed_outings_df['end_ts'])


    # --- Monthly Aggregation ---
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

    # --- Create Output Folder and Save CSVs ---
    # Utilisez le output_folder basé sur le PARTICIPANT_NUMBER
    try:
        os.makedirs(output_folder, exist_ok=True)
        processed_outings_df.to_csv(os.path.join(output_folder, "processed_outings_df.csv"), index=False)
        outings_daily_new.to_csv(os.path.join(output_folder, "outings_daily_activity.csv"), index=False)
        outings_monthly.to_csv(os.path.join(output_folder, "outings_monthly_activity.csv"), index=False)
        door_failure_daily_markers.to_csv(os.path.join(output_folder, "door_sensor_failure_days.csv"), index=False)
        print("CSV files saved in folder:", output_folder)
    except Exception as e:
        print(f"Error saving CSVs: {e}")

    return processed_outings_df, outings_daily_new, outings_monthly, door_failure_daily_markers


# --- Figure Creation Function ---
def create_outings_figure_1(processed_outings_data, aggregated_daily_data, monthly_data, daily_failure_markers,
                            scale, selected_month, selected_day): # Ajout du paramètre PARTICIPANT_NUMBER
    fig = go.Figure()

    # Ajout du numéro de participant au titre
    title_suffix = f" (Participant {PARTICIPANT_NUMBER})" if PARTICIPANT_NUMBER else ""

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

            df_monthly['failure_percentage'] = (df_monthly['door_failure_days_sum_monthly'] / df_monthly['date'].dt.daysinmonth) * 100
            fig.add_trace(go.Bar(x=df_monthly['month_label'], y=df_monthly['duration_hours_mean'], name="Durée moy. sortie (heures)", error_y=dict(type='data', array=df_monthly['duration_hours_sem']), marker_color=DATA1_COLOR))
            if 'door_failure_days_sum_monthly' in df_monthly.columns:
                fig.add_trace(go.Scatter(x=df_monthly['month_label'], y=df_monthly['failure_percentage'],yaxis='y2', name="Jours échec porte (%)", mode='lines+markers', line=dict(color=DATA3_COLOR, dash='dot')))

            y2_max_val = 5
            if 'failure_percentage' in df_monthly.columns and pd.notna(df_monthly['failure_percentage'].max()):
                y2_max_val = max(5, df_monthly['failure_percentage'].max() * 1.1)

            fig.update_layout(
                title=dict(text=f"Activité Sorties : Durée moyenne et Jours échec capteur (Vue Annuelle){title_suffix}", x=TITLE_X, y=TITLE_Y),
                xaxis=dict(title=dict(text="Mois")),
                yaxis=dict(title=dict(text="Durée moyenne sortie (heures)"), tickfont=dict(color=DATA1_COLOR), range=[0, 105]),
                yaxis2=dict(title=dict(text="Jours échec porte (%)"), tickfont=dict(color=DATA3_COLOR), showgrid=False, range=[0,y2_max_val]),
                legend=LEGEND,
                hovermode='x unified'
            )
        else:
            fig.update_layout(title=dict(text=f"Outings: No yearly data available{title_suffix}"))

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
        if not df_daily_failure_filtered.empty:
            fig.add_trace(go.Scatter(x=df_daily_failure_filtered['date'], y=[0.1] * len(df_daily_failure_filtered), name="Échec porte", mode='markers', marker=dict(color=DATA3_COLOR, size=10, symbol='x'), yaxis='y2'))

        if not df_daily_activity.empty or not df_daily_failure_filtered.empty:
            y2_range_max = 2
            fig.update_layout(
                title=dict(text=f"Activité Sorties : Durée et Échecs capteur (Vue Journalière) - {selected_month}{title_suffix}", x=TITLE_X, y=TITLE_Y),
                xaxis=dict(title=dict(text="Jour"), tickformat='%d', tickmode='array', tickvals=unique_display_dates, ticktext=[d.strftime('%d') for d in unique_display_dates] if unique_display_dates else []),
                yaxis=dict(title=dict(text="Temps dehors / jour (heures)"), range=[0, 24.5]),
                yaxis2=dict(title=dict(text="Échec porte"), range=[0, y2_range_max], showgrid=False),
                legend=LEGEND
            )
        else:
            fig.update_layout(title=dict(text=f"Outings: No daily data or door failures for {selected_month}{title_suffix}"))

    elif scale == 'month' and not selected_month:
        fig.update_layout(title=dict(text=f"Outings: Please select a month{title_suffix}"))

    elif scale == 'day' and selected_day:
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
                        title=dict(text=f"Vue Horaire : Sorties le {selected_day}{title_suffix}", x=TITLE_X, y=TITLE_Y),
                        xaxis=dict(title="Heure de la journée", tickmode='array',
                                   tickvals=list(range(24)),
                                   ticktext=[f"{h:02d}:00" for h in range(24)]),
                        yaxis=dict(title="Présence dehors (fraction de l'heure)", range=[0, 1.1]),
                        legend=LEGEND,
                        yaxis2=dict(showticklabels=False, showgrid=False, zeroline=False, title=None)
                    )
                else:
                    fig.update_layout(title=dict(text=f"Day View: No outings data for {selected_day}{title_suffix}"))
            except Exception as e:
                print(f"Error processing hourly outing view for {selected_day}: {e}")
                fig.update_layout(title=dict(text=f"Error loading hourly data for {selected_day}"))
        else:
            fig.update_layout(title=dict(text="Day View: Processed outings data not available or empty"))

    elif scale == 'day' and not selected_day:
        fig.update_layout(title=dict(text=f"Day View: Please select a day (after selecting a month){title_suffix}"))


    if not fig.data:
        fig.update_layout(title=dict(text=f"Aucune donnée à afficher pour la sélection actuelle{title_suffix}"))
    return fig

def create_outings_figure_2(processed_outings_data, aggregated_daily_data, monthly_data, daily_failure_markers,
                            scale, selected_month, selected_day): # Ajout du paramètre PARTICIPANT_NUMBER
    fig = go.Figure()

    # Ajout du numéro de participant au titre
    title_suffix = f" (Participant {PARTICIPANT_NUMBER})" if PARTICIPANT_NUMBER else ""

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
            fig.add_trace(go.Bar(x=df_monthly['month_label'], y=df_monthly['activity_count_sum'], name="Nb. sorties / mois", marker_color=DATA2_COLOR))

            y_max_val = df_monthly['activity_count_sum'].max() if pd.notna(df_monthly['activity_count_sum'].max()) else 5
            y_range_max = max(3, y_max_val * 1.1 if y_max_val > 0 else 3)

            fig.update_layout(
                title=dict(text=f"Activité Sorties : Nombre de sorties (Vue Annuelle){title_suffix}", x=TITLE_X, y=TITLE_Y),
                xaxis=dict(title=dict(text="Mois")),
                yaxis=dict(title=dict(text="Nb. sorties / mois"), tickfont=dict(color=DATA2_COLOR), range=[0, y_range_max]),
                legend=LEGEND
            )
        else:
            fig.update_layout(title=dict(text=f"Outings: No yearly data available{title_suffix}"))

    elif scale == 'month' and selected_month:
        df_daily_activity = aggregated_daily_data[aggregated_daily_data['year_month'] == selected_month] if not aggregated_daily_data.empty else pd.DataFrame()

        if not df_daily_activity.empty:
            fig.add_trace(go.Bar(x=df_daily_activity['date'], y=df_daily_activity['activity_count_sum'], name="Nb. sorties terminées / jour", marker_color=DATA2_COLOR))

        if not df_daily_activity.empty:
            all_relevant_dates = df_daily_activity['date'].tolist()
            unique_display_dates = sorted(list(set(all_relevant_dates))) if all_relevant_dates else []

            y_max = df_daily_activity['activity_count_sum'].max() if not df_daily_activity.empty else 0
            y_range_max = max(3, y_max * 1.1 if pd.notna(y_max) else 3)

            fig.update_layout(
                title=dict(text=f"Activité Sorties : Nombre de sorties (Vue Journalière) - {selected_month}{title_suffix}", x=TITLE_X, y=TITLE_Y),
                xaxis=dict(title=dict(text="Jour"), tickformat='%d', tickmode='array', tickvals=unique_display_dates, ticktext=[d.strftime('%d') for d in unique_display_dates] if unique_display_dates else []),
                yaxis=dict(title=dict(text="Nb. sorties terminées"), range=[0, y_range_max]),
                legend=LEGEND
            )
        else:
            fig.update_layout(title=dict(text=f"Outings: No daily data for {selected_month}{title_suffix}"))

    elif scale == 'day' and selected_day:
        fig.update_layout(title=dict(text=f"Day View: Outing count is not directly applicable for hourly view{title_suffix}"))

    if not fig.data:
        fig.update_layout(title=dict(text=f"Aucune donnée à afficher pour la sélection actuelle{title_suffix}"))
    return fig

