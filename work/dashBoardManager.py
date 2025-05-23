import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State, no_update
import numpy as np
import os

# --- Global Configuration ---
TEXT_COLOR = 'white'
BACKGROUND_COLOR = '#111111'

TOILET_APP_NAME = "Toilet Activity"
SLEEP_APP_NAME = "Sleep Activity"
OUTINGS_APP_NAME = "Outings Activity"

# Determine available participant folders
def get_available_participants():
    participants = []
    current_dir = os.path.dirname(__file__)
    for item in os.listdir(current_dir):
        if item.startswith('participant_') and os.path.isdir(os.path.join(current_dir, item)):
            try:
                participant_num = int(item.split('_')[1])
                participants.append(participant_num)
            except ValueError:
                continue
    return sorted(participants)

AVAILABLE_PARTICIPANTS = get_available_participants()
DEFAULT_PARTICIPANT = AVAILABLE_PARTICIPANTS[0] if AVAILABLE_PARTICIPANTS else 1

print(f"Available participants: {AVAILABLE_PARTICIPANTS}")

# --- Data Processing Utility ---
def process_daily_data(df):
    if df.empty:
        for col in ['timestamp', 'year_month', 'date', 'hour']:
            if col not in df.columns:
                df[col] = None
        return df

    if 'timestamp' not in df.columns:
        if 'year_month' in df.columns and 'day' in df.columns:
            try:
                df['date_str_temp'] = df['year_month'] + '-' + df['day'].astype(str).str.zfill(2)
                df['timestamp'] = pd.to_datetime(df['date_str_temp'], errors='coerce')
                df.drop(columns=['date_str_temp'], inplace=True)
            except Exception:
                df['timestamp'] = pd.NaT
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            df['timestamp'] = pd.NaT

    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        if 'year_month' not in df.columns:
            df['year_month'] = df['timestamp'].dt.strftime('%Y-%m')
        if 'date' not in df.columns:
            df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
        if 'hour' not in df.columns:
            df['hour'] = df['timestamp'].dt.hour
    else:
        for col, default_val in [('year_month', None), ('date', None), ('hour', None)]:
            if col not in df.columns:
                df[col] = default_val
    return df

# Initialize a dictionary to hold data for each participant
PARTICIPANT_DATA = {}

# Stubs for functions - they will now accept participant_number
try:
    from visualize_toilet_new import get_toilet_data, create_toilet_figure_1, create_toilet_figure_2
    print("Successfully imported toilet functions.")
except ImportError as e:
    print(f"Error importing from visualize_toilet_new.py: {e}")
    def get_toilet_data(participant_number): return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    def create_toilet_figure_1(*args): return go.Figure().update_layout(title_text="Error loading toilet module (stub)")
    def create_toilet_figure_2(*args): return go.Figure().update_layout(title_text="Error loading toilet module (stub)")

try:
    from visualize_sleep_quiet_new import get_sleep_data, create_sleep_figure
    print("Successfully imported sleep functions.")
except ImportError as e:
    print(f"Error importing from visualize_sleep_quiet_new.py: {e}")
    def get_sleep_data(participant_number): return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    def create_sleep_figure(*args): return go.Figure().update_layout(title_text="Error loading sleep module (stub)")

try:
    from visualize_outing_new import get_outings_data, create_outings_figure_1, create_outings_figure_2
    print("Successfully imported outings functions.")
except ImportError as e:
    print(f"Error importing from visualize_outing_new.py: {e}")
    def get_outings_data(participant_number): return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    def create_outings_figure_1(*args): return go.Figure().update_layout(title_text="Error loading outings module (stub)")
    def create_outings_figure_2(*args): return go.Figure().update_layout(title_text="Error loading outings module (stub)")

# --- Dash App Initialization ---
app = Dash(__name__)
app.title = "Activity Dashboard Manager"

# --- App Layout ---
app.layout = html.Div(id="app-container", children=[
    html.H2(app.title, id="app-title"),
    html.Div(id="control-container", children=[
        html.Div(className='selector', children=[
            html.Label("Participant :"),
            dcc.Dropdown(
                id='participant-selector',
                options=[{'label': f'Participant {p}', 'value': p} for p in AVAILABLE_PARTICIPANTS],
                value=DEFAULT_PARTICIPANT,
                clearable=False,
                className='dash-dropdown'
            ),
        ]),
        html.Div(className='selector', children=[
            html.Label("Activity :"),
            dcc.Dropdown(
                id='activity-type-selector',
                options=[
                    {'label': TOILET_APP_NAME, 'value': 'toilet'},
                    {'label': SLEEP_APP_NAME, 'value': 'sleep'},
                    {'label': OUTINGS_APP_NAME, 'value': 'outings'}
                ],
                value='toilet',
                clearable=False,
                className='dash-dropdown'
            ),
        ]),
        html.Div(className='selector', children=[
            html.Label("View scale :"),
            dcc.Dropdown(
                id='scale-selector',
                options=[
                    {'label': 'Year View', 'value': 'year'},
                    {'label': 'Month View', 'value': 'month'},
                    {'label': 'Day View', 'value': 'day'}
                ],
                value='year',
                clearable=False,
                className='dash-dropdown'
            ),
        ]),
        html.Div(id='month-dropdown-container', className='selector', children=[
            html.Label("Month:"),
            dcc.Dropdown(
                id='month-dropdown',
                clearable=False,
                className='dash-dropdown'
            )
        ], style={'display': 'none'}),
        html.Div(id='day-dropdown-container', className='selector', children=[
            html.Label("Day:"),
            dcc.Dropdown(
                id='day-dropdown',
                clearable=False,
                className='dash-dropdown'
            )
        ], style={'display': 'none'}),
    ]),
    dcc.Graph(id='activity-graph-1'),
    html.Div(id='second-chart-container', children=[
        dcc.Graph(id='activity-graph-2')
    ], style={'display': 'none'})
])

# --- Callbacks ---

@app.callback(
    Output('app-container', 'children', allow_duplicate=True),
    Input('participant-selector', 'value'),
    prevent_initial_call='demand'
)
def load_participant_data(selected_participant_number):
    if selected_participant_number not in PARTICIPANT_DATA:
        print(f"Loading data for participant {selected_participant_number}...")
        
        p_toilet_raw_ts_df, p_toilet_daily_agg_df, p_toilet_monthly_agg_df, p_toilet_failure_markers_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        p_sleep_raw_ts_df, p_sleep_daily_agg_df, p_sleep_monthly_agg_df, p_sleep_bed_failure_daily_markers = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        p_outings_raw_ts_df, p_outings_daily_agg_df, p_outings_monthly_agg_df, p_outings_door_failure_daily_markers = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        try:
            df1, df2, df3, df4 = get_toilet_data(selected_participant_number)
            p_toilet_raw_ts_df = df1.copy() if not df1.empty else pd.DataFrame()
            p_toilet_daily_agg_df = process_daily_data(df2.copy() if not df2.empty else pd.DataFrame())
            p_toilet_monthly_agg_df = df3.copy() if not df3.empty else pd.DataFrame()
            p_toilet_failure_markers_df = df4.copy() if not df4.empty else pd.DataFrame()
            print(f"Toilet data loaded for participant {selected_participant_number}: Raw shape={p_toilet_raw_ts_df.shape}")
        except ValueError as ve:
            print(f"Error unpacking data from get_toilet_data for participant {selected_participant_number}: {ve}")
        except Exception as e:
            print(f"Error running get_toilet_data for participant {selected_participant_number}: {e}")

        try:
            df1, df2, df3, df4 = get_sleep_data(selected_participant_number)
            p_sleep_raw_ts_df = df1.copy() if not df1.empty else pd.DataFrame()
            p_sleep_daily_agg_df = process_daily_data(df2.copy() if not df2.empty else pd.DataFrame())
            p_sleep_monthly_agg_df = df3.copy() if not df3.empty else pd.DataFrame()
            p_sleep_bed_failure_daily_markers = df4.copy() if not df4.empty else pd.DataFrame()
            print(f"Sleep data loaded for participant {selected_participant_number}: Raw shape={p_sleep_raw_ts_df.shape}")
        except ValueError as ve:
            print(f"Error unpacking data from get_sleep_data for participant {selected_participant_number}: {ve}")
        except Exception as e:
            print(f"Error running get_sleep_data for participant {selected_participant_number}: {e}")

        try:
            df1, df2, df3, df4 = get_outings_data(selected_participant_number)
            p_outings_raw_ts_df = df1.copy() if not df1.empty else pd.DataFrame()
            p_outings_daily_agg_df = process_daily_data(df2.copy() if not df2.empty else pd.DataFrame())
            p_outings_monthly_agg_df = df3.copy() if not df3.empty else pd.DataFrame()
            p_outings_door_failure_daily_markers = df4.copy() if not df4.empty else pd.DataFrame()
            print(f"Outings data loaded for participant {selected_participant_number}: Raw shape={p_outings_raw_ts_df.shape}")
        except ValueError as ve:
            print(f"Error unpacking data from get_outings_data for participant {selected_participant_number}: {ve}")
        except Exception as e:
            print(f"Error running get_outings_data for participant {selected_participant_number}: {e}")

        PARTICIPANT_DATA[selected_participant_number] = {
            'toilet_raw_ts_df': p_toilet_raw_ts_df,
            'toilet_daily_agg_df': p_toilet_daily_agg_df,
            'toilet_monthly_agg_df': p_toilet_monthly_agg_df,
            'toilet_failure_markers_df': p_toilet_failure_markers_df,
            'sleep_raw_ts_df': p_sleep_raw_ts_df,
            'sleep_daily_agg_df': p_sleep_daily_agg_df,
            'sleep_monthly_agg_df': p_sleep_monthly_agg_df,
            'sleep_bed_failure_daily_markers': p_sleep_bed_failure_daily_markers,
            'outings_raw_ts_df': p_outings_raw_ts_df,
            'outings_daily_agg_df': p_outings_daily_agg_df,
            'outings_monthly_agg_df': p_outings_monthly_agg_df,
            'outings_door_failure_daily_markers': p_outings_door_failure_daily_markers,
        }
        print(f"Data for participant {selected_participant_number} stored in PARTICIPANT_DATA.")
    
    return no_update

@app.callback(
    Output('month-dropdown-container', 'style'),
    Output('day-dropdown-container', 'style'),
    Input('scale-selector', 'value')
)
def toggle_dropdown_visibility(scale):
    month_style = {'display': 'none'}
    day_style = {'display': 'none'}
    if scale == 'month':
        month_style = {'display': 'flex'}
    elif scale == 'day':
        month_style = {'display': 'flex'}
        day_style = {'display': 'flex'}
    return month_style, day_style

@app.callback(
    Output('month-dropdown', 'options'),
    Output('month-dropdown', 'value'),
    Input('activity-type-selector', 'value'),
    Input('participant-selector', 'value')
)
def update_month_dropdown_options(selected_activity_type, selected_participant_number):
    options = []
    value = None
    df_daily_for_months = pd.DataFrame()

    participant_data = PARTICIPANT_DATA.get(selected_participant_number, {})

    if selected_activity_type == 'toilet':
        df_daily_for_months = participant_data.get('toilet_daily_agg_df', pd.DataFrame())
    elif selected_activity_type == 'sleep':
        df_daily_for_months = participant_data.get('sleep_daily_agg_df', pd.DataFrame())
    elif selected_activity_type == 'outings':
        df_daily_for_months = participant_data.get('outings_daily_agg_df', pd.DataFrame())

    available_months_set = set()
    if not df_daily_for_months.empty and 'year_month' in df_daily_for_months.columns:
        valid_months = df_daily_for_months['year_month'].dropna()
        if not valid_months.empty:
            available_months_set.update(valid_months.unique())

    if available_months_set:
        available_months_sorted = sorted(list(available_months_set))
        options = [{'label': str(m), 'value': str(m)} for m in available_months_sorted]
        if available_months_sorted:
            value = available_months_sorted[0]
    return options, value

@app.callback(
    Output('day-dropdown', 'options'),
    Output('day-dropdown', 'value'),
    Input('activity-type-selector', 'value'),
    Input('month-dropdown', 'value'),
    Input('scale-selector', 'value'),
    Input('participant-selector', 'value')
)
def update_day_dropdown_options(selected_activity_type, selected_month, scale, selected_participant_number):
    if scale != 'day' or not selected_month:
        return [], None

    options = []
    value = None
    df_daily_for_days = pd.DataFrame()

    participant_data = PARTICIPANT_DATA.get(selected_participant_number, {})

    if selected_activity_type == 'toilet':
        df_daily_for_days = participant_data.get('toilet_daily_agg_df', pd.DataFrame())
    elif selected_activity_type == 'sleep':
        df_daily_for_days = participant_data.get('sleep_daily_agg_df', pd.DataFrame())
    elif selected_activity_type == 'outings':
        df_daily_for_days = participant_data.get('outings_daily_agg_df', pd.DataFrame())

    if not df_daily_for_days.empty and 'date' in df_daily_for_days.columns and 'year_month' in df_daily_for_days.columns:
        if selected_month and df_daily_for_days['year_month'].notna().any():

            days_in_month_df = df_daily_for_days[df_daily_for_days['year_month'] == selected_month]
        
            valid_dates = days_in_month_df['date'].dropna()
            if not valid_dates.empty:
                available_days = sorted(valid_dates.unique()) 
                options = []
                for d_str in available_days:
                    try:
                        dt_obj = pd.to_datetime(d_str)
                        label = dt_obj.strftime('%d/%m')
                        options.append({'label': label, 'value': d_str})
                    except ValueError:
                        options.append({'label': d_str, 'value': d_str})
                if available_days:
                    value = available_days[0]
    return options, value

@app.callback(
    Output('second-chart-container', 'style'),
    Input('activity-type-selector', 'value')
)
def toggle_second_chart_visibility(activity_type):
    if activity_type == 'toilet' or activity_type == 'outings':
        return {'display': 'block', 'marginTop': '20px'}
    else:
        return {'display': 'none'}

@app.callback(
    Output('activity-graph-1', 'figure'),
    Input('activity-type-selector', 'value'),
    Input('scale-selector', 'value'),
    Input('month-dropdown', 'value'),
    Input('day-dropdown', 'value'),
    Input('participant-selector', 'value')
)
def update_main_graph(activity_type, scale, selected_month, selected_day, selected_participant_number):
    fig = go.Figure()
    current_month = selected_month if scale in ['month', 'day'] else None
    current_day = selected_day if scale == 'day' else None

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR)
    )

    participant_data = PARTICIPANT_DATA.get(selected_participant_number, {})

    try:
        create_fn = None
        df1_pass, df2_pass, df3_pass, df4_pass = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        if activity_type == 'toilet':
            create_fn = create_toilet_figure_1
            df1_pass = participant_data.get('toilet_raw_ts_df', pd.DataFrame())
            df2_pass = participant_data.get('toilet_daily_agg_df', pd.DataFrame())
            df3_pass = participant_data.get('toilet_monthly_agg_df', pd.DataFrame())
            df4_pass = participant_data.get('toilet_failure_markers_df', pd.DataFrame())
        elif activity_type == 'sleep':
            create_fn = create_sleep_figure
            df1_pass = participant_data.get('sleep_raw_ts_df', pd.DataFrame())
            df2_pass = participant_data.get('sleep_daily_agg_df', pd.DataFrame())
            df3_pass = participant_data.get('sleep_monthly_agg_df', pd.DataFrame())
            df4_pass = participant_data.get('sleep_bed_failure_daily_markers', pd.DataFrame())
        elif activity_type == 'outings':
            create_fn = create_outings_figure_1
            df1_pass = participant_data.get('outings_raw_ts_df', pd.DataFrame())
            df2_pass = participant_data.get('outings_daily_agg_df', pd.DataFrame())
            df3_pass = participant_data.get('outings_monthly_agg_df', pd.DataFrame())
            df4_pass = participant_data.get('outings_door_failure_daily_markers', pd.DataFrame())
        
        if create_fn:
            fig = create_fn(
                df1_pass,    
                df2_pass,    
                df3_pass,  
                df4_pass,    
                scale,
                current_month,
                current_day
            )
        else:
            fig.update_layout(title=dict(text="Select an activity type", font=dict(color=TEXT_COLOR)))
            
    except Exception as e:
        print(f"Error creating figure for {activity_type} with scale {scale} for participant {selected_participant_number}: {e}")
        fig.update_layout(
            title=dict(text=f"Error generating graph for {activity_type}: Check console", font=dict(color='red'))
        )

    current_title = fig.layout.title.text if fig.layout and fig.layout.title and fig.layout.title.text else ""
    is_stub_title = "Error loading module (stub)" in current_title

    if not fig.data and (not current_title or is_stub_title):
        title_text = "Veuillez sélectionner une activité et une échelle."
        if scale == 'day':
            if not current_month:
                title_text = "Veuillez d'abord sélectionner un mois."
            elif not current_day:
                title_text = "Veuillez sélectionner un jour pour la vue horaire."
        elif scale == 'month' and not current_month:
            title_text = "Veuillez sélectionner un mois pour la vue journalière."
        
        if is_stub_title and not fig.data :
            fig.update_layout(title_text = title_text)
        elif not fig.data :
            fig.update_layout(title_text = title_text)

    fig.update_layout(
        template='plotly_dark', paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR)
    )
    if "Error generating graph" in current_title:
        fig.update_layout(title=dict(text=current_title, font=dict(color='red')))

    return fig

@app.callback(
    Output('activity-graph-2', 'figure'),
    Input('activity-type-selector', 'value'),
    Input('scale-selector', 'value'),
    Input('month-dropdown', 'value'),
    Input('day-dropdown', 'value'),
    Input('participant-selector', 'value')
)
def update_second_graph(activity_type, scale, selected_month, selected_day, selected_participant_number):
    fig = go.Figure()
    current_month = selected_month if scale in ['month', 'day'] else None
    current_day = selected_day if scale == 'day' else None

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR)
    )

    participant_data = PARTICIPANT_DATA.get(selected_participant_number, {})

    if activity_type == 'toilet':
        try:
            fig = create_toilet_figure_2(
                participant_data.get('toilet_raw_ts_df', pd.DataFrame()),    
                participant_data.get('toilet_daily_agg_df', pd.DataFrame()),    
                participant_data.get('toilet_monthly_agg_df', pd.DataFrame()),    
                participant_data.get('toilet_failure_markers_df', pd.DataFrame()),  
                scale,
                current_month,
                current_day
            )
        except Exception as e:
            print(f"Error creating second figure for toilet with scale {scale} for participant {selected_participant_number}: {e}")
            fig.update_layout(
                title=dict(text=f"Error generating second graph: Check console", font=dict(color='red'))
            )
    elif activity_type == 'outings':
        try:
            fig = create_outings_figure_2(
                participant_data.get('outings_raw_ts_df', pd.DataFrame()),
                participant_data.get('outings_daily_agg_df', pd.DataFrame()),
                participant_data.get('outings_monthly_agg_df', pd.DataFrame()),
                participant_data.get('outings_door_failure_daily_markers', pd.DataFrame()),
                scale,
                current_month,
                current_day
            )
        except Exception as e:
            print(f"Error creating second figure for outings with scale {scale} for participant {selected_participant_number}: {e}")
            fig.update_layout(
                title=dict(text=f"Error generating second graph: Check console", font=dict(color='red'))
            )
    else:
        fig.update_layout(title=dict(text="No second chart available", font=dict(color=TEXT_COLOR)))

    current_title = fig.layout.title.text if fig.layout and fig.layout.title and fig.layout.title.text else ""
    is_stub_title = "Error loading module (stub)" in current_title

    if not fig.data and (not current_title or is_stub_title):
        title_text = "Veuillez sélectionner une activité et une échelle."
        if scale == 'day':
            if not current_month:
                title_text = "Veuillez d'abord sélectionner un mois."
            elif not current_day:
                title_text = "Veuillez sélectionner un jour pour la vue horaire."
        elif scale == 'month' and not current_month:
            title_text = "Veuillez sélectionner un mois pour la vue journalière."
        
        if is_stub_title and not fig.data:
            fig.update_layout(title_text=title_text)
        elif not fig.data:
            fig.update_layout(title_text=title_text)

    return fig

# --- App Execution ---
if __name__ == '__main__':
    print("Starting Dash Manager App...")
    app.run(debug=True)