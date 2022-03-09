import bisect
import collections.abc
import math
import numbers

from collections import deque
import datetime

import numpy as np
import warnings

from copy import deepcopy

# loot existing functionality
from covid19sim.utils.mobility_planner import Activity, ACTIVITIES, MobilityPlanner,\
    _human_dies, _presample_activity, _patch_schedule, _sample_activity_duration, _can_accept_invite,\
    _modify_schedule, _can_supervise_kid, _reallocate_residence, _select_location, \
    _get_datetime_for_seconds_since_midnight


from covid19sim.utils.utils import _random_choice, filter_queue_max, filter_open, compute_distance, _normalize_scores, _get_seconds_since_midnight, log
from covid19sim.utils.constants import SECONDS_PER_DAY, SECONDS_PER_HOUR, SECONDS_PER_MINUTE

class InteractivePlanner(MobilityPlanner):
    """
    Scheduler planning object that prepares `human`s schedule from the time of waking up to sleeping on the same day.

    Args:
        human (covid19sim.human.Human): `human` for whom this schedule needs to be planned
        env (simpy.Environment): simpy environment that schedules these `activities`
        conf (dict): yaml configuration of the experiment
    """
    def __init__(self, human, env, conf):
        super().__init__(human, env, conf)

    def __repr__(self):
        return f"<MobilityPlanner for {self.human}>"

    def initialize(self):
        """
        Initializes current activity to be sleeping until AVG_SLEEPING_MINUTES.
        Prepares a tentative schedule for the entire simulation so that only the location needs to be determined.
        Following cases are considered -
            1. `human` is a kid that can't be without parent supervision
            2. `human` is a kid that can go to school, but needs parent supervision at other times
            3. `human` who is free to do anything.
        """
        # start human from the activity of sleeping. (assuming everyone sleeps for same amount of time)
        AVERAGE_TIME_SLEEPING = self.conf['AVERAGE_TIME_SLEEPING']
        duration = AVERAGE_TIME_SLEEPING * SECONDS_PER_HOUR

        self.full_schedule = RangeDict()
        self.full_schedule[(int(self.env.timestamp.timestamp()), int(self.env.timestamp.timestamp())+duration)] = Activity(self.env.timestamp, duration, "sleep", self.human.household, self.human)
        self.current_activity = self.full_schedule[int(self.env.timestamp.timestamp())]

        # presample activities for the entire simulation
        # simulation is run until these many days pass. We want to sample for all of these days. Add 1 to include the activities on the last day.
        # Add an additional 1 to be on teh safe side and sample activities for an extra day.
        n_days = self.conf['simulation_days'] + 1
        todays_weekday = self.env.timestamp.weekday()

        MAX_AGE_CHILDREN_WITHOUT_SUPERVISION = self.conf['MAX_AGE_CHILDREN_WITHOUT_PARENT_SUPERVISION']
        if self.human.age <= MAX_AGE_CHILDREN_WITHOUT_SUPERVISION:
            self.follows_adult_schedule = True
            self.adults_in_house = [h for h in self.human.household.residents if h.age > MAX_AGE_CHILDREN_WITHOUT_SUPERVISION]
            if len(self.adults_in_house) > 0:
                self.adult_to_follow_today = self.rng.choice(self.adults_in_house, size=1).item()
                self.adult_to_follow_today.mobility_planner.inverted_supervision.add(self.human)
            else:
                self.follows_adult_schedule = False
                log(f"Improper housing allocation has led to {self.human} living without adult. MobilityPlanner will not keep them supervised.", self.human.city.logfile)

        if not self.follows_adult_schedule:
            ## work
            if self.human.does_not_work:
                does_work = np.zeros(n_days)
            else:
                does_work = 1.0 * np.array([(todays_weekday + i) % 7 in self.human.working_days for i in range(n_days)])
                n_working_days = (does_work > 0).sum()
                does_work[does_work > 0] = [_sample_activity_duration("work", self.conf, self.rng) for _ in range(n_working_days)]

            ## other activities
            does_grocery = _presample_activity("grocery", self.conf, self.rng, n_days)
            does_exercise = _presample_activity("exercise", self.conf, self.rng, n_days)
            does_socialize = _presample_activity("socialize", self.conf, self.rng, n_days)

            # schedule them all while satisfying sleep constraints
            # Note: we sample locations on the day of activity
            last_activity = self.current_activity
            full_schedule = []
            for i in range(n_days):
                assert last_activity.name == "sleep", f"found {last_activity} and not sleep"

                # Note: order of appending is important to _patch_schedule
                # Note: duration of activities is equally important. A variance factor of 10 in the distribution
                # might result in duration spanning two or more days which will violate the assumptions in this planner.
                to_schedule = []
                tentative_date = (self.env.timestamp + datetime.timedelta(days=i)).date()

                #debug
                if self.human.name == 'human:481':
                    print("DEBUG: glitch agent")

                to_schedule.append(Activity(None, does_work[i].item(), "work", self.human.workplace, self.human, tentative_date))
                to_schedule.append(Activity(None, does_socialize[i].item(), "socialize", None, self.human, tentative_date))
                to_schedule.append(Activity(None, does_grocery[i].item(), "grocery", None, self.human, tentative_date))
                to_schedule.append(Activity(None, does_exercise[i].item(), "exercise", None, self.human, tentative_date))

                # adds idle and sleep acivities too
                schedule = _patch_schedule(self.human, last_activity, to_schedule, self.conf)
                last_activity = schedule[-1]
                full_schedule.append(schedule)
                # (debug)
                if last_activity.duration == 0:
                    warnings.warn(f"{self.human} has 0 duration {last_activity}\nschedule:{schedule}\npenultimate:{full_schedule[-2]}")

            assert all(schedule[-1].name == "sleep" for schedule in full_schedule), "sleep not found as last element in a schedule"
            assert len(full_schedule) == n_days, "not enough schedule prepared"

            # fill the schedule with sleep if there is some time left at the end
            time_left_to_simulation_end = (full_schedule[-1][-1].end_time -  self.env.timestamp).total_seconds()
            assert time_left_to_simulation_end > SECONDS_PER_DAY, "A full day's schedule has not been planned"
            if time_left_to_simulation_end < n_days * SECONDS_PER_DAY:
                filler_schedule = deque([Activity(full_schedule[-1][-1].end_time, time_left_to_simulation_end, "sleep", self.human.household, self.human, prepend_name="filler")])
                full_schedule.append(filler_schedule)

            # interesting part, lets make this work a bit no?
            for day in full_schedule:
                for activity in day:
                    self.full_schedule[(int(activity.start_time.timestamp()), int(activity.end_time.timestamp()))] = activity
            today = int(self.env.timestamp.timestamp())
            self.schedule_for_day = self.full_schedule.get_range(today, today+SECONDS_PER_DAY)
            self.current_activity = self.schedule_for_day.pop(0)

    def get_schedule(self, for_kids=False):
        """
        Moves the schedule pointer to the schedule for the current simulation day.

        Args:
            force_range (start_time, end_time): returns schedule that spans over all the acitivities across start_time and end_time
            force_end_time (datetime.datetime): return all the activities until first "sleep" which have end_time greater than force_end_time
        Returns:
            schedule (deque): a deque of `Activity`s where the activities are arranged in increasing order of their starting time.
        """
        today = self.env.timestamp.date()

        if for_kids:
            assert not self.follows_adult_schedule, "kids do not have preplanned schedule"
            # on the last simulation day, at the time of this function call, adult might not have next_schedule.
            next_schedule = []
            if len(self.full_schedule.get_range(int(self.current_activity.end_time.timestamp()))) > 0:
                next_schedule = self.full_schedule.get_range(int(self.current_activity.end_time.timestamp()), int(today.timestamp()+SECONDS_PER_DAY))
            return [self.current_activity] + self.schedule_for_day + next_schedule

        if len(self.schedule_for_day) == 0:
            self.schedule_for_day = self._prepare_schedule()
            self.schedule_prepared.add(today)

        return self.schedule_for_day

    def get_next_activity(self):
        """
        Clears `schedule_for_day` by popping the last element and storing it in `current_activity`.
        Also calls `prepare_schedule` when there are no more activities.

        Returns:
            (Activity): activity that human does next
        """
        schedule = self.get_schedule()
        self.current_activity = schedule.pop(0)
        self.current_activity = self._modify_activity_location_if_needed(self.current_activity)
        return self.current_activity

    def receive(self, activity):
        """
        Receives the invite from `human`.

        Args:
            activity (Activity): attributes of activity

        Returns:
            (bool): True if `self` adds `activity` to its schedule. False o.w.
        """
        assert activity.name == "socialize", "coordination for other activities is not implemented."
        today = self.env.timestamp.date()

        current_schedule = [self.current_activity] + self.schedule_for_day
        if (
            not _can_accept_invite(today, self)
            or (
                # Feature NotImplmented - to Schedule an event that modifies both the current_schedule and the next_schedule
                # ignoring the equality (in second cond. of `and`) here will make `activity` to be the last `schedule_for_day` which breaks the invariant that `last_activity` in `schedule_for_day` should be sleep.
                activity.start_time < current_schedule[-1].end_time
                and activity.end_time >= current_schedule[-1].end_time
            )
            or ( # can't schedule an event before the current_event ends
                activity.start_time < self.current_activity.end_time
            )
            or (
                # can only modify remaining schedule (sleep) if currently human is not sleeping (not waking up early)
                self.current_activity.name == "sleep"
                and activity.end_time <= current_schedule[-1].end_time
            )
        ):
            return False

        self.invitation["received"].add(today)

        P_INVITATION_ACCEPTANCE = self.conf['P_INVITATION_ACCEPTANCE']
        if self.rng.random() < 1 - P_INVITATION_ACCEPTANCE:
            return False

        # invitations are sent on the day of the event
        # only accept this activity if it fits in the schedule of the day on which it is sent
        # and leave the current schedule unchanged

        update_next_schedule = False
        next_schedule = self.full_schedule.get_range(today.timestamp(), today.timestamp() + SECONDS_PER_DAY)
        # find schedule such that the invitation activity ends before the schedule ends
        # that schedule needs to be updated
        if activity.end_time <= current_schedule[-1].end_time:
            # its a double check wrt to the above condition (kept it here for better readability)
            if self.current_activity.name == "sleep":
                return False
            else:
                remaining_schedule = [self.current_activity]
                new_schedule = self.schedule_for_day
        else:
            update_next_schedule = True
            remaining_schedule = [self.current_activity] + self.schedule_for_day # this is only [current_activity] if current_activity.name == "sleep"
            new_schedule = self.full_schedule.get_range(today.timestamp(), today.timestamp() + SECONDS_PER_DAY)

        # /!\ by accepting the invite, `self` doesn't invite others to its social
        # thus, if there is a non-overlapping social on the schedule, `self` will go alone.
        new_activity = activity.clone(prepend_name="invitation", new_owner=self.human)
        new_activity.set_location_tracker(activity)

        new_schedule, valid = _modify_schedule(self.human, remaining_schedule, new_activity, new_schedule)
        if valid:
            self.invitation["accepted"].add(today)
            for na in new_schedule:
                try:
                    self.full_schedule[(na.start_time.timestamp(), na.end_time.timestamp)]
                except Exception:
                    del self.full_schedule[na.start_time.timestamp()]
                    self.full_schedule[na.start_time.timestamp(), na.end_time.timestamp()] = na

            if not update_next_schedule:
                self.schedule_for_day = new_schedule
            return True

        return False

    def _prepare_schedule(self):
        """
        Prepares schedule for the next day. Retruns presampled schedule if its an adult.

        Returns:
            schedule (deque): a deque of `Activity`s where the activities are arranged in increasing order of their starting time.
        """
        assert len(self.schedule_for_day) == 0, "_prepare_schedule should only be called when there are no more activities in schedule_for_day"
        assert self.current_activity.name == "sleep", "_prepare_schedule should only be called if current_activity is 'sleep' "
        # if it's a kid that needs supervision, follow athe next schedule (until "sleep") of a random adult in the household
        if self.follows_adult_schedule:
            adults = _can_supervise_kid(self.adults_in_house)
            # when all adults in the house are dead --
            if len(adults) == 0:
                household = _reallocate_residence(self.human, self.human.city.households, self.rng, self.conf)
                if household is None:
                    raise NotImplementedError(f"There is no house available to supervise {self.human}")
                #
                log(f"Transferring {self.human} to {household} because no adult at {self.human.household} is alive. Current residents at {household}: {household.residents}", self.human.city.logfile)
                index_case_history = self.human.household.remove_resident(self.human)
                self.human.assign_household(household)
                household.add_resident(self.human, index_case_history)
                #
                MAX_AGE_CHILDREN_WITHOUT_SUPERVISION = self.conf['MAX_AGE_CHILDREN_WITHOUT_PARENT_SUPERVISION']
                self.adults_in_house = [h for h in self.human.household.residents if h.age > MAX_AGE_CHILDREN_WITHOUT_SUPERVISION]
                adults = self._can_supervise_kid(self.adults_in_house)


            adult = self.rng.choice(adults, size=1).item()
            adult_schedule = adult.mobility_planner.get_schedule(for_kids = True)

            work_activity = None
            if not self.human.does_not_work and self.env.timestamp.weekday() in self.human.working_days:
                work = _sample_activity_duration("work", self.conf, self.rng)
                work_activity = Activity(None, work, "work", self.human.workplace, self.human, self.env.timestamp.date())

            schedule = _patch_kid_schedule(self.human, adult_schedule, work_activity, self.current_activity, self.conf)

            # set up leader and follower through inverted_supervision
            # kid follows adult's location all the time except for when kid is hospitalized or has to stay_at_home
            # adult checks inverted_supervision everytime before finalizing the location
            # thus, if kid needs to be followed, adult follows kid to their location
            self.adult_to_follow_today.mobility_planner.inverted_supervision.remove(self.human)
            self.adult_to_follow_today = adult
            self.adult_to_follow_today.mobility_planner.inverted_supervision.add(self.human)

        else:
            today = self.env.timestamp.date()
            schedule = self.full_schedule.get_range(today.timestamp(), today.timestamp() + SECONDS_PER_DAY)
            self.schedule_day += 1

        return schedule

    def send_social_invites(self):
        """
        Sends invitation for "socialize" activity to `self.human.known_connections`.
        NOTE (IMPORTANT): To be called once per day at midnight. By calling it at midnight helps in modifying schedules of those who accept the invitation.
        """
        today = self.env.timestamp.date()

        # invite others
        if self._can_send_invite(today, self):
            todays_activities = []
            all_activities = [self.current_activity] + list(self.schedule_for_day)
            all_activities += [] if len(self.full_schedule.get_range(today.timestamp() + SECONDS_PER_DAY, today.timestamp() + SECONDS_PER_DAY*2)) == 0 else list(self.full_schedule.get_range(today.timestamp(), today.timestamp() + SECONDS_PER_DAY))
            for activity in all_activities:
                if activity.start_time.day == today.day or activity.end_time.day == today.day:
                    todays_activities.append(activity)

            MIN_DURATION = min(self.conf['INFECTION_DURATION'], self.conf['MIN_MESSAGE_PASSING_DURATION'])
            socials = [x for x in todays_activities if x.name == "socialize"]
            assert len(socials) <=1, "more than one socials on one day are not allowed in preplanned scheduling"
            if socials and socials[0].duration >= MIN_DURATION:
                self.invitation["sent"].add(today)
                self.invite(socials[0], self.human.known_connections)

    def _modify_activity_location_if_needed(self, activity):
        """
        Modifies the location of `activity` (and schedule if hospitalized) given a certain condition.

        Args:
            activity (Activity): current_activity for which checks are asked for

        Returns:
            activity (Activity): current_activity with updated location based on `self.human`'s condition
        """
        assert self.death_timestamp is None, "processing activities for a dead human"

        # for `human`s dying because of `human.never_recovers` we check the flag here
        if self.human_dies_in_next_activity:
            self, human, activity = _human_dies(self, self.human, activity, self.env)
            # print(self.human,  "is dead because never recovers", activity)
            return activity

        #
        self.human.intervened_behavior.quarantine.reset_if_its_time()

        # (a) set back to normal routine if self.human was hospitalized
        # (b) if human is still recovering in hospital then return the activity as it is because these were determined at the time when hospitalization occured
        # Note 1: after recovery from hospitalization, infection_timestamp will always be None
        # Note 2: hospitalization_recovery_timestamp will take into account hospitalization recovery due to critical condition
        # Note 3: patient dies in hospital so while recovering it is necessary to check for `human_will_die_if_critical`
        if (
            (self.hospitalization_timestamp is not None
            or self.critical_condition_timestamp is not None)
            and not self.human_will_die_if_critical
            and self.env.timestamp >= self.hospitalization_recovery_timestamp
        ):
            self.hospitalization_timestamp = None
            self.critical_condition_timestamp = None
            self.human.check_covid_recovery()
            # Hospital can discharge patient once the symptoms are not severe. patient needs to be careful after treatment to not infect others and practice isolation for some time
            # assert self.human.infection_timestamp is None, f"{self.human} is out of hospital and still has COVID"

        elif (
            (self.hospitalization_timestamp is not None
            or self.critical_condition_timestamp is not None)
            and self.env.timestamp < self.hospitalization_recovery_timestamp
        ):
            return activity # while in hospital, these activities are predetermined until hospitalization recovery time

        AVERAGE_TIME_TO_HOSPITAL_GIVEN_SYMPTOMS = self.conf['AVERAGE_DAYS_TO_HOSPITAL_GIVEN_SYMPTOMS']
        AVERAGE_TIME_TO_CRITICAL_IF_HOSPITALIZED = self.conf['AVERAGE_DAYS_TO_CRITICAL_IF_HOSPITALIZED']
        AVERAGE_TIME_DEATH_IF_CRITICAL = self.conf['AVERAGE_DAYS_DEATH_IF_CRITICAL']

        # hospitalization related checks
        # Note: because of the wide variance we have used averages only
        # TODO - P - Find out which distribution these parameters belong to.
        # 1. hospitalized given symptoms
        if (
            self.human_will_be_hospitalized
            and self.human.infection_timestamp is not None
            and self.human.covid_symptom_start_time is not None
            and self.hospitalization_timestamp is None
            and (self.env.timestamp - self.human.covid_symptom_start_time).total_seconds() >= AVERAGE_TIME_TO_HOSPITAL_GIVEN_SYMPTOMS * SECONDS_PER_DAY
        ):
            self.human.city.tracker.track_hospitalization(self.human) # track
            self.hospitalization_timestamp = self.env.timestamp
            hospital = _select_location(self.human, "hospital", self.human.city, self.rng, self.conf)
            if hospital is None:
                self, human, activity = _human_dies(self, self.human, activity, self.env)
                # print(self.human,  "died because of the lack of hospital capacity")
                return activity

            activity, self = _move_relevant_activities_to_hospital(self.human, self, activity, self.rng, self.conf, hospital, critical=False)
            # print(self.human,  "is hospitalized", activity)
            return activity

        # 2. critical given hospitalized
        # self.human is moved from hospital to its ICU
        if (
            self.human_will_be_critical_if_hospitalized
            and self.human.infection_timestamp is not None
            and self.hospitalization_timestamp is not None
            and self.critical_condition_timestamp is None
            and (self.env.timestamp - self.hospitalization_timestamp).total_seconds() >= AVERAGE_TIME_TO_CRITICAL_IF_HOSPITALIZED * SECONDS_PER_DAY
        ):
            self.human.city.tracker.track_hospitalization(self.human, "icu") # track
            self.critical_condition_timestamp = self.env.timestamp
            ICU = self._select_location(self.human, "hospital-icu", self.human.city, self.rng, self.conf)
            if ICU is None:
                self, human, activity = _human_dies(self, self.human, activity, self.env)
                # print(self.human,  "died because of the lack of ICU capacity")
                return activity

            activity, self = _move_relevant_activities_to_hospital(self.human, self, activity, self.rng, self.conf, ICU, critical=True)
            # print(self.human,  "is critical", activity)
            return activity

        # 3. death given critical
        if (
            self.human_will_die_if_critical
            and self.human.infection_timestamp is not None
            and self.hospitalization_timestamp is not None
            and self.critical_condition_timestamp is not None
            and self.death_timestamp is None
            and (self.env.timestamp - self.critical_condition_timestamp).total_seconds() >= AVERAGE_TIME_DEATH_IF_CRITICAL * SECONDS_PER_DAY
        ):
            self, human, activity = _human_dies(self, self.human, activity, self.env)
            print(self.human,  "is dead because of the critical condition", activity)
            return activity

        # 4. for adults, if there is a kid that needs supervision because the kid has to stay at home or is hospitalized,
        # /!\ It doesn't let a single adult attend to two kids: one in hospital and another in house or 3 kids: in different hospitals etc..
        for kid in self.inverted_supervision:
            assert not self.follows_adult_schedule, "a kid should not go into inverted supervision"
            if (
                kid.mobility_planner.location_of_hospitalization is not None
                or kid.mobility_planner.hospitalization_timestamp is not None
                or kid.mobility_planner.critical_condition_timestamp is not None
            ):
                location = kid.mobility_planner.location_of_hospitalization # in hospitalization, kid's activities have location
                reason = "inverted-supervision-hospitalization"
                activity.cancel_and_go_to_location(reason=reason, location=location) # Note: this activity can't be supervised
                # print(self.human, activity, "for hospitalization of", kid)
                return activity

            # if kid is quarantined
            if kid.intervened_behavior.is_quarantining():
                location = kid.household
                reason = "inverted-supervision-quarantined"
                activity.cancel_and_go_to_location(reason=reason, location=location)
                return activity

            # if activity is already at home - no need to update
            # /!\ if for some reason, this condition is not valid while the kid's activity was canceled due to sickness
            # then this activity will call refresh_location and raise AssertionError due to prepend_name
            kid_to_stay_at_home = kid.mobility_planner.human_to_rest_at_home
            if (
                kid_to_stay_at_home
                and activity.location is not None
                and activity.location != kid.household
            ):
                location = kid.household #
                reason = "inverted-supervision-stay-at-home"
                activity.cancel_and_go_to_location(reason=reason, location=location)
                # print(self.human, activity, "for sick", kid)
                return activity

        # 5. if human is quarantined
        activity, quarantined = self._intervention_related_behavior_changes(activity)
        if quarantined:
            return activity

        # 6. health / intervention related checks; set the location to household
        # rest_at_home needs to be checked everytime. It is different from hospitalization which is for prespecified period of time
        rest_at_home = self._update_rest_at_home()
        if (
            rest_at_home
            and activity.location is not None
            and activity.location != self.human.household
        ):
            activity.cancel_and_go_to_location(reason="sick-rest-at_home", location=self.human.household)
            # print(self.human,  "is sick", activity)
            return activity

        # 7. (a) check the status of the parent_pointer (it has to be the invitation or supervised activity)
        # otherwise (b) find a location. if there is no location available, cancel the activity and stay at home
        if activity.parent_activity_pointer is not None:
            activity.refresh_location()
            return activity

        if activity.location is None:
            activity.location = _select_location(self.human, activity.name, self.human.city, self.rng, self.conf)
            if activity.location is None:
                activity.cancel_and_go_to_location(reason="no-location", location=self.human.household)
                print(self.human,  "can't get a location", activity)

        return activity

    def cancel_all_events(self):
        """
        Empties the remaining schedule and removes human from the residence.
        """
        self.schedule_for_day = []
        self.current_activity = None
        activities = self.full_schedule.get_range(self.env.timestamp.timestamp())
        for a in activities:
            del self.full_schedule[a.start_time.timestamp(), a.end_time.timestamp()]

def _move_relevant_activities_to_hospital(human, mobility_planner, current_activity, rng, conf, hospital, critical=False):
    """
    Changes the schedule so that `human`s future activities are at a hospital.
    Note 1: [IMPORTANT]: human.recovery_days and mobility_planner.location_of_hospitalization is updated here too
    If the recovery time after being critical is less than the previous hospitalized recovery time, then activities are put back to normal routine.

    Note 2: This function should be called only at the beginning of hospitalzation because it also discharges and admits the patient
    """
    # modify the schedule until the recovery time
    AVERAGE_DAYS_RECOVERY = conf['AVERAGE_DAYS_RECOVERY_IF_HOSPITALIZED']
    if critical:
        AVERAGE_DAYS_RECOVERY = conf['AVERAGE_DAYS_RECOVERY_IF_CRITICAL']

    AVERAGE_TIME_RECOVERY = AVERAGE_DAYS_RECOVERY * SECONDS_PER_DAY
    recovery_time = human.env.timestamp + datetime.timedelta(seconds=AVERAGE_TIME_RECOVERY)
    mobility_planner.hospitalization_recovery_timestamp = recovery_time
    human.recovery_days = AVERAGE_DAYS_RECOVERY + (human.env.timestamp - human.infection_timestamp).total_seconds() / SECONDS_PER_DAY

    # admit this human to the list of patients at the hospital/ICU
    # this assumes that call to this function is only at the beginning and the end of hospitalization / ICU
    if mobility_planner.location_of_hospitalization is not None:
        mobility_planner.location_of_hospitalization.discharge(human)
    hospital.admit_patient(human, until=recovery_time)
    mobility_planner.location_of_hospitalization = hospital

    acitivities_to_revert_back_to_normal = [] # if critical, change in recovery time will need previously modified activities to change back to normal
    activities_to_modify = []
    for activity in [current_activity] + list(mobility_planner.schedule_for_day):
        if activity.end_time < recovery_time:
            activities_to_modify.append(activity)


    for schedule in [mobility_planner.full_schedule.get_range(human.env.start_time+SECONDS_PER_DAY*d, human.env.start_time+SECONDS_PER_DAY*(d+1)) for d in range(human.env.simulation_day, human.city.conf['simulation_days'])]:
        for activity in schedule:
            if (
                activity.start_time > recovery_time
                and "Hospitalized" in activity.append_name
            ):
                acitivities_to_revert_back_to_normal.append(activity)

            elif (
                activity.start_time > recovery_time
                and "Hospitalized" not in activity.append_name
            ):
                break

            activities_to_modify.append(activity)

    for activity in activities_to_modify:
        reason = "Hospitalized" if not critical else "ICU"
        activity.cancel_and_go_to_location(reason=reason, location=hospital)

    for activity in acitivities_to_revert_back_to_normal:
        activity._add_to_append_name("-recovered")
        activity.location = None
        if activity.name == "work":
            activity.location = human.workplace

    return current_activity, mobility_planner

def _patch_kid_schedule(human, adult_schedule, work_activity, current_activity, conf):
    """
    Makes a schedule that copies `Activity`s in `adult_schedule` to the new schedule aligned with `current_activity` (expects "sleep")
    Adds work activity if a non-zero `work` is provided.

    Args:
        human (covid19sim.human.Human): human for which `activities`  needs to be scheduled
        adult_schedule (list): list of `Activity`s which have been presampled for adult for the next day
        current_activity (Activity): activity that `human` is currently doing (expects "sleep")
        work_activity (Activity): Activity of type school for kid if that kid goes to school. None otherwise.
        conf (dict): yaml configuration of the experiment

    Returns:
        schedule (deque): a deque of `Activity`s that are in continuation with `remaining_schedule`. It doesn't include `remaining_schedule`.
    """
    assert current_activity.name == "sleep", "sleep not found as the last activity"
    assert adult_schedule[-1].name == "sleep", "adult_schedule doesn't have sleep as its last element"

    last_sleep_activity = current_activity
    last_activity = current_activity
    assert all(activity.end_time > last_activity.start_time for activity in adult_schedule), "adult activities which spans kid's current activity are expected"
    assert any(activity.start_time > last_activity.end_time for activity in adult_schedule), "at least one adult activity that ends after kid's current activity is expected"

    max_awake_duration = _sample_activity_duration("awake", conf, human.rng)
    schedule, awake_duration = [], 0
    # add work just after the current_activity on the remaining_schedule
    if (
        work_activity is not None
        and human.mobility_planner.hospitalization_timestamp is None
        and human.mobility_planner.critical_condition_timestamp is None
        and human.mobility_planner.death_timestamp is None
    ):
        work_activity.start_time = _get_datetime_for_seconds_since_midnight(human.work_start_time, work_activity.tentative_date)
        # adjust activities if there is a conflict while keeping the last_activity unchanged because it's should be the current_activity of the kid
        if work_activity.start_time < last_activity.end_time:
            work_activity, last_activity = _make_hard_changes_to_activity_for_scheduling(work_activity, last_activity, keep_last_unchanged=True)

        schedule, last_activity, awake_duration = _add_to_the_schedule(human, schedule, work_activity, last_activity, awake_duration)

    candidate_activities = [activity for activity in adult_schedule if activity.end_time > last_activity.end_time]
    # 1. discard activities which are completely a subset of schedule upto now
    # 2. align the activity which has a partial overlap with current_activity
    # 3. add rest of them as it is
    for activity in candidate_activities:

        # 1.
        if activity.end_time <= last_activity.end_time:
            continue

        # 2.
        elif activity.start_time < last_activity.end_time < activity.end_time:
            new_activity = activity.align(last_activity, cut_left=True, prepend_name="supervised", new_owner=human)

        # 3.
        else:
            new_activity = activity.clone(prepend_name="supervised", new_owner=human)

        new_activity.set_location_tracker(activity) # this way location of this activity is decided only at the time of activity
        schedule, last_activity, awake_duration = _add_to_the_schedule(human, schedule, new_activity, last_activity, awake_duration)

        if awake_duration > max_awake_duration:
            break

    # finally, close the schedule by adding sleep
    schedule, last_activity, awake_duration = _add_sleep_to_schedule(human, schedule, last_sleep_activity, last_activity, human.rng, conf, awake_duration, max_awake_duration=max_awake_duration)

    # if kid is hospitalized, all the activities should take place at the hospital
    if human.mobility_planner.location_of_hospitalization is not None:
        for activity in schedule:
            activity.location = human.mobility_planner.location_of_hospitalization
            activity._add_to_append_name("-patched-hospitalized")

    full_schedule = [current_activity] + schedule
    for a1, a2 in zip(full_schedule, full_schedule[1:]):
        assert a1.end_time == a2.start_time, "times do not align"

    return deque(schedule)

def _add_to_the_schedule(human, schedule, activity, last_activity, awake_duration):
    """
    Adds `activity` to the `schedule`. Also adds "idle" `Activity` if there is a time gap between `activity` and `last_activity`.

    Args:
        human (covid19sim.human.Human): human for which `activity` needs to be added to the schedule
        schedule (list): list of `Activity`s
        activity (Activity): new `activity` that needs to be added to the `schedule`
        last_activity (Activity): last activity that `human` was doing
        awake_duration (float): total amount of time in seconds that `human` had been awake

    Returns:
        schedule (list): list of `Activity`s with the last `Activity` as sleep
        last_activity (Activity): sleep as the last activity
        awake_duration (float): total amount of time in seconds that `human` has been awake after adding the new `activity`.

    """
    assert activity.start_time is not None, "only fully defined activities are expected"
    assert activity.start_time >= last_activity.end_time, "function assumes no confilict with the last activity"

    # ** A ** # set up the activity so that it is in accordance to the previous activity and the location's opening and closing constraints

    # opening and closing time for the location of this activity
    opening_time, closing_time = _get_open_close_times(activity.name, human.conf, activity.location)

    ## check the constraints with respect to a location
    seconds_since_midnight = _get_seconds_since_midnight(activity.start_time)
    if seconds_since_midnight > closing_time:
        return schedule, last_activity, awake_duration

    if seconds_since_midnight < opening_time:
        activity.start_time = _get_datetime_for_seconds_since_midnight(opening_time, activity.tentative_date)
        if activity.start_time < last_activity.end_time:
            return schedule, last_activity, awake_duration

    # if it is not an all time open location, end_in_seconds can not exceed closing time
    if closing_time != SECONDS_PER_DAY:
        activity.duration = min(closing_time - seconds_since_midnight, activity.duration)

    if activity.duration == 0:
        # print(human, f"has {activity} of 0 duration")
        pass
    assert activity.duration >= 0, f"negative duration {activity.duration} encountered"

    # ** B ** # Add an idle activity if there is a time gap between this activity and the last activity
    idle_time = (activity.start_time - last_activity.end_time).total_seconds()

    assert idle_time >= 0, f"negative idle_time {idle_time} encountered"

    if idle_time > 0:
        schedule, last_activity, awake_duration = _add_idle_activity(human, schedule, activity, last_activity, awake_duration)

    assert last_activity.end_time == activity.start_time, f"times do not align for {last_activity} and {activity}"

    schedule.append(activity)
    return schedule, activity, awake_duration + activity.duration

def _add_sleep_to_schedule(human, schedule, last_sleep_activity, last_activity, rng, conf, awake_duration, max_awake_duration=0):
    """
    Adds sleep `Activity` to the schedule. We constrain everyone to have an awake duration during which they
    hop from one network to the other, and a sleep during which they are constrained to be at their respective household networks.

    Args:
        human (covid19sim.human.Human): human for which sleep schedule needs to be added.
        schedule (list): list of `Activity`s
        wake_up_time_in_seconds (float): seconds since midnight when `human` wake up
        last_activity (Activity): last activity that `human` was doing
        rng (np.random.RandomState): Random number generator
        conf (dict): yaml configuration of the experiment
        awake_duration (float): total amount of time in seconds that `human` had been awake

    Returns:
        schedule (list): list of `Activity`s with the last `Activity` as sleep
        last_activity (Activity): sleep as the last activity
        total_duration (float): total amount of time in seconds that `human` had spent across all the activities in the schedule.
    """
    # draw time for which `self` remains awake and sleeps
    if max_awake_duration <= 0:
        max_awake_duration = _sample_activity_duration("awake", conf, rng)
    sleep_duration = _sample_activity_duration("sleep", conf, rng)

    start_time = last_sleep_activity.end_time + datetime.timedelta(seconds=max_awake_duration)
    sleep_activity = Activity(start_time, sleep_duration, "sleep", human.household, human)

    if sleep_activity.start_time >= last_activity.end_time:
        schedule, last_activity, awake_duration = _add_idle_activity(human, schedule, sleep_activity, last_activity, awake_duration)
        return _add_to_the_schedule(human, schedule, sleep_activity, last_activity, awake_duration)

    sleep_activity, last_activity = _make_hard_changes_to_activity_for_scheduling(sleep_activity, last_activity)
    return _add_to_the_schedule(human, schedule, sleep_activity, last_activity, awake_duration)

def _make_hard_changes_to_activity_for_scheduling(next_activity, last_activity, keep_last_unchanged=False):
    """
    Makes changes to either of the activities if next_activity starts before last_activity.

    Args:
        next_activity (Activity): activity that is in conflict with the last activity
        last_activity (Activity): previous activity that starts after `next_activity`
        keep_last_unchanged (bool): if True, doesn't affect last_activity

    Returns:
        next_activity (Activity): activity in alignment with `last_activity`
        last_activity (Activity): activity in alignment with `next_activity`
    """
    def _assert_positive_duration(next_activity, last_activity):
        assert next_activity.duration >= 0 and last_activity.duration >= 0, "negative duration encountered"
        return next_activity, last_activity

    # print(f"making hard changes between next- {next_activity} and last - {last_activity}")
    # 1. if last activity can be safely cut short, do that and leave next_activity unchanged
    if not keep_last_unchanged and last_activity.start_time <= next_activity.start_time:
        last_activity.duration = (next_activity.start_time - last_activity.start_time).total_seconds()
        return _assert_positive_duration(next_activity, last_activity)

    # 2. else cut short next_activity
    # short version -
    # next_activity.start_time = last_activity.end_time
    # next_activity.duration = min(0, (next_activity.end_time - next_activity.start_time).total_seconds())
    # return next_activity, last_activity

    # more explicit
    # 2a. do next_activity late
    if next_activity.end_time >= last_activity.end_time:
        next_activity.start_time = last_activity.end_time
        next_activity.duration = (next_activity.end_time - next_activity.start_time).total_seconds()
        return _assert_positive_duration(next_activity, last_activity)

    # 2b. next_activity was supposed to end before the last activity, hence don't do next_activity
    next_activity.start_time = last_activity.end_time
    next_activity.duration = 0
    return _assert_positive_duration(next_activity, last_activity)

def _add_idle_activity(human, schedule, next_activity, last_activity, awake_duration):
    """
    Adds an idle activity at household.

    Args:
        human (covid19sim.human.Human): human for which "idle" activity needs to be added to the schedule
        schedule (list): list of `Activity`s
        next_activity (Activity): new `activity` that needs to be added to the `schedule`
        last_activity (Activity): last activity that `human` was doing
        awake_duration (float): total amount of time in seconds that `human` had been awake

    Returns:
        schedule (list): list of `Activity`s with the last `Activity` as sleep
        last_activity (Activity): sleep as the last activity
        awake_duration (float): total amount of time in seconds that `human` has been awake after adding the new `activity`.
    """
    duration = (next_activity.start_time -  last_activity.end_time).total_seconds()
    if duration == 0:
        return schedule, last_activity, awake_duration

    assert duration > 0, "negative duration for idle activity is not allowed"
    idle_activity = Activity(last_activity.end_time, duration, "idle", human.household, human)

    idle_time = (idle_activity.end_time - next_activity.start_time).total_seconds()
    assert idle_time == 0,  "non-zero idle time after adding idle_activity"

    schedule.append(idle_activity)
    return schedule, idle_activity, awake_duration + duration

def _get_open_close_times(activity_name, conf, location=None):
    """
    Fetches opening and closing time for a `location` (if given) or a typical location where `activity_name` can take place.

    Args:
        name (str): type of activity
        location (covid19sim.locations.Location): location for which opening closing times are requested.
        conf (dict): yaml configuration of the experiment

    Returns:
        opening_time (float): opening time in seconds since midnight of `activity.location`
        closing_time (float): closing time in seconds since midnight of `activity.location`
    """
    if location is not None:
        return location.opening_time, location.closing_time

    elif activity_name == "grocery":
        location_type = "STORE"
    elif activity_name == "socialize":
        location_type = "MISC"
    elif activity_name == "exercise":
        location_type = "PARK"
    elif activity_name in ["sleep", "idle"]:
        location_type = "HOUSEHOLD"
    else:
        raise ValueError(f"Unknown activity_name:{activity_name}")

    # # /!\ same calculation is in covid19sim.locations.location.Location.__init__()
    OPEN_CLOSE_TIMES = conf[f'{location_type}_OPEN_CLOSE_HOUR_MINUTE']
    opening_time = OPEN_CLOSE_TIMES[0][0] * SECONDS_PER_HOUR +  OPEN_CLOSE_TIMES[0][1] * SECONDS_PER_MINUTE
    closing_time = OPEN_CLOSE_TIMES[1][0] * SECONDS_PER_HOUR +  OPEN_CLOSE_TIMES[1][1] * SECONDS_PER_MINUTE

    return opening_time, closing_time

# =================
# ranged dict class
# =================
def _binary_key_search(LUT, key):
    """
    binary search through the LUT, assumes forward LUT.
    """
    Lower = 0
    Upper = len(LUT)
    while Lower <= Upper:
        i = math.floor((Lower + Upper) / 2)  # get middle index
        if i == len(LUT):
            break
        if LUT[i][1] <= key:
            Lower = i + 1
        elif LUT[i][0] > key:
            Upper = i - 1
        elif LUT[i][0] <= key < LUT[i][1]:
            return i
    raise RangDictKeyException("Key not found")


class RangeDict(collections.abc.MutableMapping):
    """
    This datastructure takes a minimum and max as a key, and maps it to a given value.
    """

    def __init__(self, *args, **kwargs):
        self._store = dict()
        self._FLUT = []  # key (start, end)
        self._BLUT = []  # key (end, start); used purely to double check that the given addition is valid, inherent override rejection.
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key: numbers.Number or (numbers.Number, numbers.Number)):
        # grab the nearest range. I this should be O(log n) or O(1) depending on the key used
        if key is tuple and key in self._store:
            return self._store[key]

        i = _binary_key_search(self._FLUT, key)
        bounds = self._FLUT[i]
        # test that we are in range.
        return self._store[self._FLUT[i]]

    def __setitem__(self, key: (numbers.Number, numbers.Number), value):
        # (O(n)) insert step, I wish it were faster, but to get fast lookup takes time.
        self._check_range(key)
        bisect.insort_left(self._FLUT, (key[0], key[1]))
        bisect.insort_left(self._BLUT, (key[1], key[0]))
        self._store[key] = value

    def __delitem__(self, key: numbers.Number or (numbers.Number, numbers.Number)):
        # if we are given a tuple of an object to set, then grab the start time.
        if key is tuple:
            key = key[0]
        # again, time complexity O(n), the random access really hurts us here.
        i = _binary_key_search(self._FLUT, key)
        del self._store[self._FLUT[i]]
        del self._FLUT[i]
        del self._BLUT[i]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def _check_range(self, key: (numbers.Number, numbers.Number)):
        '''
        attempt to assign the key, can fail if range is of bad format, or if overlaps with an existing range
        '''
        if key[0] >= key[1]:
            raise RangDictKeyException("Range is not valid, should be (min, max); must be different")

        # forward check
        i_f = bisect.bisect_left(self._FLUT, (key[0], key[0]))
        i_b = bisect.bisect_left(self._BLUT, (key[1], key[1]))
        # test that we have empty lookup tables (initializing table)
        if not self._FLUT and not self._BLUT:
            return
        # test that the relative indices match (should work for all LUT size > 0)
        if i_f == i_b:
            # case: at the start
            if i_f == 0 and self._FLUT[i_f][0] < key[1]:
                raise RangDictKeyException(
                    "Range is not unique (ending of new block impinges upon existing start of a block).")
            # case: at the end
            elif i_f == len(self._FLUT) and self._FLUT[-1][1] > key[0]:
                raise RangDictKeyException("Range is not unique (start of new block is within the previous block).")
            # case: at the middle
            elif (i_f != len(self._FLUT) and i_f != 0) and (
                    self._FLUT[i_f - 1][1] > key[0] or self._FLUT[i_f][0] < key[1]):
                raise RangDictKeyException("Range is not unique (block is contained in different block).")
            # all tests passed.
            else:
                return
        raise RangDictKeyException("LUT index mismatch (should remain in sync).")

    def get_range(self, start: numbers.Number, end: numbers.Number = None):
        start_index = _binary_key_search(self._FLUT, start)
        end_index = None
        try:
            end_index = _binary_key_search(self._FLUT, end)+1
        except RangDictKeyException:
            pass
        keys = [k for k in self._FLUT[start_index:end_index]]
        return [self._store[k] for k in keys]



class RangDictKeyException(Exception):
    pass
