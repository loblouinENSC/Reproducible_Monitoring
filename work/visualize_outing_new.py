import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import numpy as np

# --- Configuration ---
OUTINGS_LOG_FILE = 'rule-outing.csv'
DOOR_FAILURE_LOG_FILE = 'days-door_failure_1week.csv' 
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
    """Loads and preprocesses outings and door failure data."""
    try:
        activity = pd.read_csv(OUTINGS_LOG_FILE, delimiter=';', decimal=",",
                               names=["date", "annotation", "activity_count", "duration"],
                               parse_dates=["date"], index_col="date")
        # Convert duration from seconds to minutes
        activity['durationMin'] = activity['duration'] / 60
    except FileNotFoundError:
        print(f"Error: '{OUTINGS_LOG_FILE}' not found.")
        activity = pd.DataFrame(columns=["annotation", "activity_count", "duration", "durationMin"],
                                index=pd.to_datetime([]))
        activity.index.name = 'date'
    except Exception as e:
        print(f"Error loading {OUTINGS_LOG_FILE}: {e}")
        activity = pd.DataFrame(columns=["annotation", "activity_count", "duration", "durationMin"],
                                index=pd.to_datetime([]))
        activity.index.name = 'date'

    try:
        door_failure = pd.read_csv(DOOR_FAILURE_LOG_FILE, delimiter=';', decimal=",",
                                   names=["date", "door_failure"],
                                   parse_dates=["date"], index_col="date")
        # Ensure door_failure is numeric, coerce errors to NaN, fill NaN with 0
        door_failure['door_failure'] = pd.to_numeric(door_failure['door_failure'], errors='coerce').fillna(0)
        # Filter for actual failures and make a copy for daily markers
        door_failure_daily_markers = door_failure[door_failure['door_failure'] > 0].copy().reset_index()

    except FileNotFoundError:
        print(f"Error: '{DOOR_FAILURE_LOG_FILE}' not found.")
        door_failure = pd.DataFrame(columns=["door_failure"], index=pd.to_datetime([]))
        door_failure.index.name = 'date'
        door_failure_daily_markers = pd.DataFrame(columns=['date', 'door_failure'])
    except Exception as e:
        print(f"Error loading {DOOR_FAILURE_LOG_FILE}: {e}")
        door_failure = pd.DataFrame(columns=["door_failure"], index=pd.to_datetime([]))
        door_failure.index.name = 'date'
        door_failure_daily_markers = pd.DataFrame(columns=['date', 'door_failure'])


    # Daily Aggregation (Outings)
    if not activity.empty:
        outings_daily = activity.resample('D').agg(
            activity_count_sum=('activity_count', 'sum'),
            duration_min_sum=('durationMin', 'sum'), # Sum of durations for the day
            duration_min_mean=('durationMin', 'mean'), # Mean duration of individual outings
            duration_min_sem=('durationMin', 'sem')    # SEM of individual outings
        ).reset_index()
        outings_daily['year_month'] = outings_daily['date'].dt.to_period('M').astype(str)
    else:
        outings_daily = pd.DataFrame(columns=['date', 'activity_count_sum', 'duration_min_sum', 'duration_min_mean', 'duration_min_sem', 'year_month'])

    # Monthly Aggregation (Outings & Door Failure)
    if not activity.empty:
        outings_monthly_agg = activity.resample('ME').agg(
            activity_count_sum=('activity_count', 'sum'),
            duration_min_mean=('durationMin', 'mean'), # Mean duration of outings in the month
            duration_min_sem=('durationMin', 'sem')    # SEM of outing durations in the month
        )
    else:
        outings_monthly_agg = pd.DataFrame(columns=['activity_count_sum', 'duration_min_mean', 'duration_min_sem'], index=pd.to_datetime([]))
        outings_monthly_agg.index.name = 'date'

    if not door_failure.empty:
        door_failure_monthly_agg = door_failure.fillna(0).resample('ME').agg(
            door_failure_day_count=('door_failure', lambda x: (x > 0).sum()) # Count days with failure
         )
    else:
        door_failure_monthly_agg = pd.DataFrame(columns=['door_failure_day_count'], index=pd.to_datetime([]))
        door_failure_monthly_agg.index.name = 'date'

    outings_monthly = pd.merge(outings_monthly_agg, door_failure_monthly_agg, left_index=True, right_index=True, how='outer')
    if not outings_monthly.empty:
        start_date, end_date = outings_monthly.index.min(), outings_monthly.index.max()
        if pd.notna(start_date) and pd.notna(end_date):
            full_idx = pd.date_range(start=start_date, end=end_date, freq='ME')
            outings_monthly = outings_monthly.reindex(full_idx)
        # Fill NaNs after reindex
        outings_monthly['activity_count_sum'] = outings_monthly['activity_count_sum'].fillna(0)
        outings_monthly['duration_min_mean'] = outings_monthly['duration_min_mean'].fillna(np.nan)
        outings_monthly['duration_min_sem'] = outings_monthly['duration_min_sem'].fillna(0)
        outings_monthly['door_failure_day_count'] = outings_monthly['door_failure_day_count'].fillna(0)

    outings_monthly = outings_monthly.reset_index().rename(columns={'index': 'date'})

    # Add year_month to daily door failure markers for filtering in the figure function
    if not door_failure_daily_markers.empty and 'date' in door_failure_daily_markers.columns:
        if pd.api.types.is_datetime64_any_dtype(door_failure_daily_markers['date']):
            door_failure_daily_markers['year_month'] = door_failure_daily_markers['date'].dt.to_period('M').astype(str)
        else:
            door_failure_daily_markers['date'] = pd.to_datetime(door_failure_daily_markers['date'], errors='coerce')
            door_failure_daily_markers.dropna(subset=['date'], inplace=True)
            if not door_failure_daily_markers.empty:
                door_failure_daily_markers['year_month'] = door_failure_daily_markers['date'].dt.to_period('M').astype(str)

    return outings_daily, outings_monthly, door_failure_daily_markers

# --- Figure Creation Function ---
def create_outings_figure(daily_data, monthly_data, daily_failure_markers, scale, selected_month):
    """Creates the Plotly figure for outings activity."""
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
        yaxis2=dict(title=dict(font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR), gridcolor='rgba(255, 255, 255, 0.1)', overlaying='y', side='right'),
        margin=MARGIN_CHART
    )

    if scale == 'year':
        df = monthly_data
        if not df.empty:
            # Y1: Average duration of outings
            fig.add_trace(go.Bar(
                x=df['date'],
                y=df['duration_min_mean'],
                name="Durée moy. sortie (min)",
                error_y=dict(type='data', array=df['duration_min_sem']),
                marker_color=DATA1_COLOR
            ))
            # Y2: Number of outings
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['activity_count_sum'],
                name="Nb. sorties / mois",
                yaxis='y2',
                mode='lines+markers',
                line=dict(color=DATA2_COLOR)
            ))
            # Y2: Door failure days
            if 'door_failure_day_count' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['door_failure_day_count'],
                    name="Jours échec porte / mois",
                    yaxis='y2',
                    mode='lines+markers',
                    line=dict(color=DATA3_COLOR, dash='dot')
                ))
            fig.update_layout(
                title=dict(text="Activité Sorties : Vue Annuelle (Mensuelle)", font=dict(color=TEXT_COLOR), x=TITLE_X, y=TITLE_Y),
                xaxis=dict(title=dict(text="Mois", font=dict(color=TEXT_COLOR))),
                yaxis=dict(title=dict(text="Durée moyenne sortie (min)", font=dict(color=DATA1_COLOR)), tickfont=dict(color=DATA1_COLOR)),
                yaxis2=dict(title=dict(text="Nombre", font=dict(color=DATA2_COLOR)), tickfont=dict(color=DATA2_COLOR), showgrid=False),
                legend=LEGEND
            )
        else:
            fig.update_layout(title=dict(text="Outings: No yearly data available", font=dict(color=TEXT_COLOR)))

    elif scale == 'month' and selected_month:
        df_daily_outings = daily_data[daily_data['year_month'] == selected_month]

        df_daily_fail_markers = pd.DataFrame() # Initialize empty
        if not daily_failure_markers.empty and 'year_month' in daily_failure_markers.columns:
             df_daily_fail_markers = daily_failure_markers[daily_failure_markers['year_month'] == selected_month]

        if not df_daily_outings.empty:
            # Y1: Number of outings per day
            fig.add_trace(go.Bar(
                x=df_daily_outings['date'],
                y=df_daily_outings['activity_count_sum'],
                name="Nb. sorties / jour",
                marker_color=DATA2_COLOR
            ))
            # Y2: Total duration of outings per day
            fig.add_trace(go.Scatter(
                x=df_daily_outings['date'],
                y=df_daily_outings['duration_min_sum'],
                name="Durée totale sorties / jour (min)",
                yaxis='y2',
                mode='lines',
                line=dict(color=DATA1_COLOR)
            ))
            # Markers for door failure days
            if not df_daily_fail_markers.empty:
                fig.add_trace(go.Scatter(
                    x=df_daily_fail_markers['date'],
                    y=[0.1] * len(df_daily_fail_markers), # Position near x-axis on primary y-axis
                    name="Échec porte",
                    mode='markers',
                    marker=dict(color=DATA3_COLOR, size=10, symbol='x'),
                    yaxis='y1' # Explicitly assign to y1 if y2 is used for something else
                ))

            fig.update_layout(
                title=dict(text=f"Activité Sorties : Vue Journalière - {selected_month}", font=dict(color=TEXT_COLOR), x=TITLE_X, y=TITLE_Y),
                xaxis=dict(title=dict(text="Jour", font=dict(color=TEXT_COLOR))),
                yaxis=dict(title=dict(text="Nb. sorties / jour", font=dict(color=DATA2_COLOR)), tickfont=dict(color=DATA2_COLOR)),
                yaxis2=dict(title=dict(text="Durée totale sorties / jour (min)", font=dict(color=DATA1_COLOR)), tickfont=dict(color=DATA1_COLOR), showgrid=False),
                legend=LEGEND
            )
        else:
            fig.update_layout(title=dict(text=f"Outings: No daily data for {selected_month}", font=dict(color=TEXT_COLOR)))
    elif scale == 'month' and not selected_month:
        fig.update_layout(title=dict(text="Outings: Please select a month", font=dict(color=TEXT_COLOR)))

    if not fig.data:
        fig.update_layout(title=dict(text="Aucune donnée à afficher", font=dict(color=TEXT_COLOR)))
    return fig

# --- Standalone App Execution ---
if __name__ == '__main__':
    outings_daily_data, outings_monthly_data, door_failure_daily_markers_data = get_outings_data()
    available_months = sorted(outings_daily_data['year_month'].unique()) if not outings_daily_data.empty else []

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
        return create_outings_figure(outings_daily_data, outings_monthly_data, door_failure_daily_markers_data, scale, selected_month)

    app.run(debug=True)
