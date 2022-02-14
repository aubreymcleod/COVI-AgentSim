"""
Class for storing a controlled representation of an agent and their given schedule.

It will support operations that allow the user to view properties of the agent,
tweak select parameters about the agent, and adjust activities that are on the agent's
schedule
"""
import time
import datetime
import covid19sim.utils.mobility_planner as mp

# all activities that the agent can perform
ACTIVITIES = mp.ACTIVITIES

# a shell that allows agents to be addressed by their name, which will make it easier
# handle from the perspective of a web based application
class Agent_container:
    def __init__(self, agents):
        self.agents = {agent.name:Agent_Representation(agent) for agent in agents}



class Agent_Representation:
    def __init__(self, agent, sim):
        # pointers to the raw objects this class masks.
        self._sim_ref = sim
        self._agent_ref = agent
        self._mobility_planer_ref = agent.mobility_planner

        # dict of all attributes in dict form
        self._attributes = agent.__dict__
        # dict of all publicly facing attributes
        self.registed_attributes = {}


        self._daily_schedule = self._mobility_planer_ref.schedule_for_day

        # register all modifiable_attributes
        self._register_attributes()

    def _register_attributes(self):
        pass


    def get_activities_in_range(self, start, end):
        activities = []
        # check if we are exploring the current day
        if datetime.fromtimestamp(start).date() == self._agent_ref.env.timestamp.date():
            if time.mktime(self._mobility_planer_ref.current_activity.start_time.timetuple()) >= start \
            or time.mktime(self._mobility_planer_ref.current_activity.end_time.timetuple()) <= end:
                activities.append(self._mobility_planer_ref.current_activity)

            for act in self._mobility_planer_ref.schedule_for_day:
                if time.mktime(act.start_time.timetuple()) >= start and time.mktime(act.start_time.timetuple()) <= end:
                    activities.append(act)
        #check every other day
        for day in self._mobility_planer_ref.full_schedule:
            for act in day:
                if time.mktime(act.start_time.timetuple()) >= start and time.mktime(act.start_time.timetuple()) <= end:
                    activities.append(act)

        return activities
