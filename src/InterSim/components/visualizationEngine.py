import pandas as pd
import matplotlib.pyplot as plt
import descartes
from mpl_toolkits.axisartist.axislines import Subplot
from shapely.geometry import Point, Polygon

class visualizer:
    def __init__(self, sim):
        # bells and whistles
        self.bg = None
        self.road_layer = None

        # location data
        self.school_ref = sim.city.schools
        self.hospital_ref = sim.city.hospitals
        self.park_ref = sim.city.parks
        self.store_ref = sim.city.stores
        self.workplace_ref = sim.city.workplaces
        self.senior_residence_ref = sim.city.senior_residences
        self.home_ref = [home for home in sim.city.households]

        # people data
        self.humans = {human.name : human for human in sim.city.humans}

        # config
        self.building_radius = 20.0
        self.school_colour = "blue"
        self.hospital_colour = "orange"
        self.home_colour = "green"
        self.park_colour = "yellow"
        self.senior_residence_colour = "purple"
        self.store_colour = "red"
        self.workplace_colour = "pink"

        self.person_radius = 5.0
        self.person_colour = "turquoise"
        self.child_colour = "aquamarine"

        #plot
        self.fig = plt.figure(figsize=(15, 15))
        self.ax = self.fig.add_subplot()

        self.building_df = self._init_buildings()

        self.draw()
        #self.ax.scatter(self.building_df.x, self.building_df.y, s=self.building_df.radius, c=self.building_df.colour)
        """
        self.school_layer = Subplot(self.fig, 111)
        self.fig.add_subplot(self.school_layer)

        self.hospital_layer = Subplot(self.fig, 111)
        self.fig.add_subplot(self.hospital_layer)

        self.park_layer = Subplot(self.fig, 111)
        self.fig.add_subplot(self.park_layer)

        self.store_layer = Subplot(self.fig, 111)
        self.fig.add_subplot(self.store_layer)

        self.work_layer = Subplot(self.fig, 111)
        self.fig.add_subplot(self.work_layer)

        self.senior_layer = Subplot(self.fig, 111)
        self.fig.add_subplot(self.senior_layer)

        self.home_layer = Subplot(self.fig, 111)
        self.fig.add_subplot(self.home_layer)

        #self.people_layer = self.fig.add_subplot()

        #self.draw_people()
        """

    def _compile_building(self, building_arr, colour):
        return pd.DataFrame({'x': [building.lon for building in building_arr],
                           'y': [building.lat for building in building_arr],
                           'radius': self.building_radius,
                           'colour': colour})
        #building_layer.scatter(df.x, df.y, s=self.building_radius, c=colour)

    def _init_buildings(self):
        building_set = self._compile_building(self.school_ref, self.school_colour)
        building_set = building_set.append(self._compile_building(self.hospital_ref, self.hospital_colour))
        building_set = building_set.append(self._compile_building(self.park_ref, self.park_colour))
        building_set = building_set.append(self._compile_building(self.store_ref, self.store_colour))
        building_set = building_set.append(self._compile_building(self.workplace_ref, self.workplace_colour))
        building_set = building_set.append(self._compile_building(self.senior_residence_ref, self.senior_residence_colour))
        building_set = building_set.append(self._compile_building(self.home_ref, self.home_colour))
        return building_set

    def draw(self):
        living = [human for human in self.humans.values() if not human.is_dead]
        people_set = pd.DataFrame({'x': [human.lon for human in living],
                           'y': [human.lat for human in living],
                           'radius': self.person_radius,
                           'colour': [self.child_colour if human.mobility_planner.follows_adult_schedule else self.person_colour for human in living]})
        self.ax.clear()
        joint_set = self.building_df.append(people_set)
        self.ax.scatter(joint_set.x, joint_set.y, s=joint_set.radius, c=joint_set.colour)
        #self.people_layer.remove()
        #self.fig.add_subplot()
        #for human in self.humans.values():
        #    if not human.is_dead:
        #        if human.mobility_planner.follows_adult_schedule:
        #            self.people_layer.scatter(human.lon, human.lat, s=self.person_radius, c=self.child_colour)
        #        else:
        #            self.people_layer.scatter(human.lon, human.lat, s=self.person_radius, c=self.person_colour)
