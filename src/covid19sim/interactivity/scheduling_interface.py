import datetime
from datetime import time, timedelta

from covid19sim.human import Human
from covid19sim.interactivity.interactive_planner import RangeDict
from covid19sim.locations.location import Location
from covid19sim.native import Environment
from covid19sim.utils.mobility_planner import Activity
from covid19sim.utils.utils import _get_seconds_since_midnight
from covid19sim.utils.constants import SECONDS_PER_HOUR, SECONDS_PER_MINUTE


# I make 4 assumptions with this API:
# 1. You cannot change the past
# 2. You cannot change the present
# 3. You cannot change sleep
# 4. You cannot affect a child's schedule

# =========
# utilities
# =========
def time_from_seconds(seconds_from_midnight: int):
    hours = int(seconds_from_midnight / SECONDS_PER_HOUR)
    minutes = int((seconds_from_midnight-hours*SECONDS_PER_HOUR) / SECONDS_PER_MINUTE)
    seconds = int(seconds_from_midnight % SECONDS_PER_MINUTE)
    if hours > 23:
        hours = 0
    return time(hours, minutes, seconds)

# =================
# Assumption checks
# =================
def in_past(env: Environment, activity: Activity):
    """
    is the chosen operation in the past based on the current time.
    Args:
        env: the current evironment used to get the time.
        activity: given activity object
    Returns:
        True if the activity has already happened. False if it hasn't.
    """
    return activity.end_time.timestamp() < env.timestamp.timestamp()


def in_present(env: Environment, activity: Activity):
    """
    is this the current activity based on the time.
    Args:
        env: the current evironment used to get the time.
        activity: a given activity object (tested to see if it is the current object)
    Returns:
        False if this is not the current activity, True if it is the current activity
    """
    return activity.start_time.timestamp() < env.timestamp.timestamp() < activity.end_time.timestamp()


def is_sleep(activity: Activity):
    """
    is the activity a sleep activity
    Args:
        activity: a given activity object (tested to see if it is the current object)
    Returns:
        True if it is sleep, False if not sleep.
    """
    return activity.name == "sleep"


def is_child(human: Human):
    """
    is the given human a child?
    Args:
        human: a given human
    Returns:
        True if a child, False if adult.
    """
    return human.mobility_planner.follows_adult_schedule


def can_edit(human: Human, activity: Activity):
    """
    Checks if an edit can be made to the activity based on my 3 constraints
    1. Edits must not be to the current simulated activity.
    2. Edits must not be in the past.
    3. Edits cannot adjust sleep.
    4. the human is not a child
    Args:
        human
        activity: given activity object
    Returns:
        True if all my constraints are met
    """
    if in_past(human.env, activity) or in_present(human.env, activity) or is_sleep(activity) or is_child(human):
        return False
    return True


# ====================
# Information retreval
# ====================
def get_schedule(human: Human):
    return human.mobility_planner.backing_schedule.get_range(human.env.initial_timestamp.timestamp())

def get_current_day_schedule(human: Human):
    return list(human.mobility_planner.schedule_for_day)

def get_editable_range(human: Human, activity: Activity):
    prior = human.mobility_planner.backing_schedule[(activity.start_time-timedelta(seconds=1.0)).timestamp()]
    following = human.mobility_planner.backing_schedule[activity.end_time.timestamp()]
    earliest_edit = prior.start_time if prior.name == "idle" else activity.start_time
    latest_edit = following.end_time if following.name == "idle" else activity.end_time
    return earliest_edit, latest_edit

# =============
# Minor changes
# =============
def update_location(human: Human, activity: Activity, location: Location):
    # check that we can edit this activity
    if not can_edit(human, activity):
        return False, "Could not edit activity."
    # check that the start and end times of the activity fall in times when the location is open
    activity_date = activity.start_time.date()
    if location.is_open(activity_date) and location.opening_time <= _get_seconds_since_midnight(activity.start_time.time()) and _get_seconds_since_midnight(activity.end_time.time()) <= location.closing_time:
        activity.location = location
        return True, "Successfully updated location."
    else:
        return False, "Could not reassign location because location would not be open at that time."


def cancel_activity(human: Human, activity: Activity):
    if not can_edit(human, activity):
        return False, "Could not cancel activity."
    activity.cancel_and_go_to_location(reason="USER_CANCELLED")
    return True, "Activity has been cancelled."


# ===============
# Complex changes
# ===============
def adjust_times(human: Human, activity: Activity, start_time: datetime, end_time: datetime, bounds: (datetime, datetime)):
    # case move start_time and maintain end_time
    adjust_start_time(human, activity, new_start_time=bounds[0])
    adjust_end_time(human, activity, new_end_time=bounds[1])
    adjust_start_time(human, activity, new_start_time=start_time)
    adjust_end_time(human, activity, new_end_time=end_time)
    return True, "Successfully made edit"



def adjust_start_time(human: Human, activity: Activity, new_start_time: datetime):
    if not can_edit(human, activity):
        return False, "Can not edit activity's start time"
    if new_start_time >= activity.end_time:
        return False, "Can not move start_time such that it surpasses the end_time."
    if new_start_time == activity.start_time:
        return False, "start_time does not change. No edit made."

    # case 1: start_time is pushed back.
    if activity.start_time < new_start_time:
        preceeding_activity = human.mobility_planner.backing_schedule.get_range(activity.start_time.timestamp()-1.0, activity.end_time.timestamp())[0]
        edits = []

        if preceeding_activity.name == "idle" and preceeding_activity != human.mobility_planner.current_activity:
            preceeding_activity = preceeding_activity.clone()
            preceeding_activity.adjust_time((new_start_time-activity.start_time).total_seconds(), False)
            edits.append(preceeding_activity)

        else:
            edits.append(Activity(activity.start_time, (new_start_time-activity.start_time).total_seconds(), "idle", human.household, human))

        new_activity = activity
        new_activity.adjust_time((new_start_time-activity.start_time).total_seconds())
        edits.append(new_activity)
        human.mobility_planner.update_schedule(edits)
        return True, "Successfully pushed start time back"

    # case 2: start_time is pushed forward.
    else:
        coverage = human.mobility_planner.backing_schedule.get_range(new_start_time.timestamp(), activity.end_time.timestamp())
        act_index = None
        for index, item in enumerate(coverage):
            if item == activity:
                act_index = index
                break

        if coverage[act_index-1].name == "idle" and coverage[act_index-1] != human.mobility_planner.current_activity:
            edits = []
            if coverage[act_index-1].start_time >= new_start_time:
                activity = activity
                activity.adjust_time(-(activity.start_time-coverage[act_index-1].start_time).total_seconds())
                edits.append(activity)
            else:
                activity = activity
                activity.adjust_time((activity.start_time-new_start_time).total_seconds())
                edits.append(Activity(coverage[act_index-1].start_time, (coverage[act_index-1].end_time-new_start_time).total_seconds(), "idle", human.household, human))
                edits.append(activity)
            human.mobility_planner.update_schedule(edits)
            return True, "Successfully pushed start time forward"
        else:
            return False, "Could not overwrite existing activity"


def adjust_end_time(human: Human, activity: Activity, new_end_time: datetime):
    if not can_edit(human, activity):
        return False, "Can not edit activity's end time"
    if new_end_time <= activity.start_time:
        return False, "Can not move end_time such that it predates the start_time."
    if new_end_time == activity.end_time:
        return False, "end_time does not change. No edit made."

    # case 1: end_time is pushed backwards.
    if activity.end_time < new_end_time:
        following_activity = human.mobility_planner.backing_schedule.get_range(activity.start_time.timestamp(), activity.end_time.timestamp())[-1]
        edits = []


        if following_activity.name == "idle":
            if new_end_time >= following_activity.end_time:
                activity = activity
                activity.adjust_time((following_activity.end_time-activity.end_time).total_seconds(), False)
                edits.append(activity)
            else:
                activity = activity
                following_activity = following_activity.clone()
                delta = (new_end_time-activity.end_time).total_seconds()
                activity.adjust_time(delta, False)
                following_activity.adjust_time(delta, True)
                edits.append(activity)
                edits.append(following_activity)
            human.mobility_planner.update_schedule(edits)
            return True, "Successfully pushed end time backwards"
        else:
            return False, "Cannot push end into non-idle time"

    # case 2: end_time is pushed forward.
    else:
        following_activity = human.mobility_planner.backing_schedule.get_range(activity.end_time.timestamp(), activity.end_time.timestamp()+1.0)[-1]
        edits = []
        if following_activity.name == "idle":
            activity = activity
            delta = (new_end_time-activity.end_time).total_seconds()
            activity.adjust_time(delta, False)
            edits.append(activity)
            following_activity = following_activity.clone()
            following_activity.adjust_time(delta, True)
            edits.append(following_activity)
        else:
            activity = activity
            following_activity = Activity(new_end_time, (activity.end_time-new_end_time).total_seconds(), "idle", human.household, human)
            activity.adjust_time((new_end_time-activity.end_time).total_seconds(), False)
            edits.append(activity)
            edits.append(following_activity)
        human.mobility_planner.update_schedule(edits)
        return True, "Successfully pushed end time forwards"


def delete_activity(human: Human, activity: Activity):
    if not can_edit(human, activity):
        return False, "Could not delete activity."
    if activity.name == "idle":
        return False, "Cannot delete idle time."
    activity.cancel_and_go_to_location(reason="USER_DELETION_FLAG")
    temporary_start = activity.start_time
    temporary_end = activity.end_time

    # see if we need to merge idle time.
    previous_activity = human.mobility_planner.backing_schedule[activity.start_time.timestamp()-1.0]
    next_activity = human.mobility_planner.backing_schedule[activity.end_time.timestamp()]

    if previous_activity != human.mobility_planner.current_activity and previous_activity.name == "idle":
        temporary_start = previous_activity.start_time
    if next_activity.name == "idle":
        temporary_end = next_activity.end_time

    human.mobility_planner.update_schedule([Activity(temporary_start, (temporary_end-temporary_start).total_seconds(), "idle", human.household, human)])
    return True, "Successfully deleted activity"


def insert_activity(human: Human, activity: Activity):
    # test that we are accessing only idle time.
    existing_schedule = human.mobility_planner.backing_schedule.get_range(activity.start_time.timestamp(), activity.end_time.timestamp()-1.0)
    for item in existing_schedule:
        if item.name != "idle":
            return False, "Trying to write into non-idle time."
        if not can_edit(human, item):
            return False, "Unable to insert the activity; failed edit checks."

    edits = []
    if existing_schedule[0].start_time < activity.start_time:
        edits.append(Activity(existing_schedule[0].start_time, (activity.start_time-existing_schedule[0].start_time).total_seconds(), "idle", human.household, human))

    edits.append(activity)

    if existing_schedule[-1].end_time > activity.end_time:
        edits.append(Activity(activity.end_time, (existing_schedule[-1].end_time-activity.end_time).total_seconds(), "idle", human.household, human))
    human.mobility_planner.update_schedule(edits)
    return True, "Successfully inserted activity"
