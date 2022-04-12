from datetime import datetime, timedelta
import streamlit as st

from covid19sim.interactivity.interactive_planner import ACTIVITIES, Activity
import covid19sim.interactivity.scheduling_interface as si

from InterSim.main import plotly_config

timesteps = {"1 second": 1.0,
             "10 seconds": 10.0,
             "1 minute": 60.0,
             "10 minutes": 600.0,
             "1 hour": 3600.0,
             "4 hours": 14400.0,
             "12 hours": 43200.0,
             "24 hours": 86400.0}

weekday_labels = {0: "Monday",
                  1: "Tuesday",
                  2: "Wednesday",
                  3: "Thursday",
                  4: "Friday",
                  5: "Saturday",
                  6: "Sunday"}

def _get_time_range(window_bounds):
    selection_range = {(window_bounds[0]+timedelta(seconds=i)).time() : (window_bounds[0]+timedelta(seconds=i)) for i in range((window_bounds[1] - window_bounds[0]).seconds+1)}
    return selection_range

def draw(plot_ref):
    # time control box.
    with st.form("Time_control", clear_on_submit=True):
        with st.sidebar:
            st.write("Simulation Date")
            timestamp = st.empty()
            timestamp.info(f'{weekday_labels[st.session_state["sim"].env.timestamp.weekday()]}  \n {st.session_state["sim"].env.timestamp}')
            speed_selection = st.select_slider("stepsize", options=timesteps.keys())
            if st.form_submit_button('Step forward'):
                if st.session_state["sim"].env.now == st.session_state["sim"].end_time:
                    pass
                    msg = "Could not advance time, we are at the end of the simulation."
                    # toastr pop fail message
                else:
                    with st.spinner(text="Please wait..."):
                        msg = st.session_state["sim"].auto_step(timesteps[speed_selection])
                        # toastr pop the msg
                    timestamp.info(f'{weekday_labels[st.session_state["sim"].env.timestamp.weekday()]}  \n {st.session_state["sim"].env.timestamp}')
                    st.session_state['renderer'].draw()
                    plot_ref.plotly_chart(st.session_state['renderer'].fig, use_container_width=True, config=plotly_config)

    # agent display widget
    agent_control_widget()

def agent_control_widget():
    with st.sidebar:
        active_agent_name = st.selectbox(label="Agent",
                                    options=[valid.name for valid in st.session_state["sim"].people.collection.values() if not valid.is_dead and not valid.mobility_planner.follows_adult_schedule],
                                    index=0)
        aa = st.session_state["sim"].people.collection[active_agent_name]
        #display agents current information
        st.info(f'{active_agent_name}  \n '
                f'Currently: {aa.mobility_planner.current_activity.name}@{aa.mobility_planner.current_activity.location.name}  \n '
                f'From: {aa.mobility_planner.current_activity.start_time}  \n '
                f'To: {aa.mobility_planner.current_activity.end_time}  \n '
                f'Lon: {aa.lon}, Lat: {aa.lat}')


        #editing pane
        st.header("Edit events")
        schedule = {activity.name+"@"+str(activity.start_time.date())+"  \n "
                                +str(activity.start_time.time())+"-"+str(activity.end_time.time()) : activity for activity in si.get_schedule(aa) if activity.start_time >= st.session_state["sim"].env.timestamp and activity.name != "idle" and activity.name != "sleep"}
        schedule_selection = st.selectbox(label="Daily activites",
                                      options=schedule.keys(),
                                      index=0)  # and activity.start_time < datetime.fromtimestamp(st.session_state["sim"].env.now + timesteps["24 hours"])

        if schedule_selection is not None:
            #change location pane
            current_location = schedule[schedule_selection].location.name if schedule_selection is not None and schedule[schedule_selection].location is not None else None
            if current_location in st.session_state["sim"].locations.keys():
                index = list(st.session_state["sim"].locations.keys()).index(current_location)
            else:
                index = 0

            st.write(f'Location set {current_location if current_location is not None else "Nowhere"}')
            location_selection = st.selectbox(label="Location",
                                                options=st.session_state["sim"].locations.keys(),
                                                index=index)
            st.write(f'Open Between: {si.time_from_seconds(st.session_state["sim"].locations[location_selection].opening_time)}-{si.time_from_seconds(st.session_state["sim"].locations[location_selection].closing_time)}  \n '
                     f'Open on: {[weekday_labels[wd] for wd in st.session_state["sim"].locations[location_selection].open_days]}')
            if st.button("Update Location"):
                status, message = si.update_location(aa, schedule[schedule_selection], st.session_state["sim"].locations[location_selection])
                print(message)

            #change timing pane
            window_bounds = si.get_editable_range(aa, schedule[schedule_selection])
            window = _get_time_range(window_bounds)
            new_start, new_end = st.select_slider("Bounds",
                                                    options=window,
                                                    value=(schedule[schedule_selection].start_time.time(), schedule[schedule_selection].end_time.time()))
            if st.button("Update Timing"):
                status, message = si.adjust_times(aa, schedule[schedule_selection], window[new_start], window[new_end], (list(window.values())[0], list(window.values())[-1]))
                print(message)

            # delete event
            if st.button("Delete Event"):
                status, message = si.delete_activity(human=aa, activity=schedule[schedule_selection])
                print(message)

        else:
            st.info("No Activity available: schedule is completely empty")

        # insert new event
        st.header("Insert New Event")
        activity_type = st.selectbox(label="Event Type",
                                     options=ACTIVITIES,
                                     index=0)
        open_slots = {activity.name + "@" + str(activity.start_time.date()) + "  \n "
                      + str(activity.start_time.time()) + "-" + str(activity.end_time.time()): activity for activity
                      in si.get_schedule(aa) if
                      activity.start_time >= st.session_state["sim"].env.timestamp and activity.name == "idle"}
        slot_selection = st.selectbox(label="Open Timeslots",
                                      options=open_slots.keys(),
                                      index=0)
        if st.button("Insert Event"):
            # start_time, duration, name, location, owner,
            activity = Activity(open_slots[slot_selection].start_time, open_slots[slot_selection].duration,
                                activity_type, aa.household, aa)
            status, message = si.insert_activity(aa, activity)
            print(message)