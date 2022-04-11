"""
Class for storing a controlled representation of an agent and their given schedule.

It will support operations that allow the user to view properties of the agent,
tweak select parameters about the agent, and adjust activities that are on the agent's
schedule
"""
import time
import datetime
import covid19sim.interactivity.scheduling_interface as si
import covid19sim.interactivity.interactive_planner as mp

# all activities that the agent can perform
ACTIVITIES = mp.ACTIVITIES

class PeopleManager:
    def __init__(self, humans):
        self.collection = {human.name : human for human in humans}

