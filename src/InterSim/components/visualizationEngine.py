import pandas as pd
import matplotlib.pyplot as plt
import descartes
import random as random
from mpl_toolkits.axisartist.axislines import Subplot
from shapely.geometry import Point, Polygon
import plotly.express as px

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
        self.misc_ref = [misc for misc in sim.city.miscs]

        # people data
        self.humans = {human.name : human for human in sim.city.humans}

        # config
        self.building_radius = 7.0
        self.person_radius = 1.0

        #plot
        self.fig = None#go.Figure()#plt.figure(figsize=(15, 15))
        #self.ax = self.fig.add_subplot()

        self.building_df = self._init_buildings()
        self.draw()
        #self.draw()
        #self.ax.scatter(self.building_df.x, self.building_df.y, s=self.building_df.radius, c=self.building_df.colour)

    def _compile_building(self, building_arr, type):
        return pd.DataFrame({'x': [building.lon for building in building_arr],
                           'y': [building.lat for building in building_arr],
                           'radius': [self.building_radius for building in building_arr],
                           'name': [building.name for building in building_arr],
                           'type': [type for building in building_arr]})
        #building_layer.scatter(df.x, df.y, s=self.building_radius, c=colour)

    def _init_buildings(self):
        building_set = self._compile_building(self.school_ref, "school")
        building_set = building_set.append(self._compile_building(self.hospital_ref, "hospital"))
        building_set = building_set.append(self._compile_building(self.park_ref, "park"))
        building_set = building_set.append(self._compile_building(self.store_ref, "store"))
        building_set = building_set.append(self._compile_building(self.workplace_ref, "workplace"))
        building_set = building_set.append(self._compile_building(self.senior_residence_ref, "senior_residence"))
        building_set = building_set.append(self._compile_building(self.home_ref, "home"))
        building_set = building_set.append(self._compile_building(self.misc_ref, "misc"))
        return building_set

    def draw(self):
        living = [human for human in self.humans.values() if not human.is_dead]
        people_set = pd.DataFrame({'x': [float(human.lon)+random.uniform(-1.0, 1.0) for human in living],
                           'y': [float(human.lat)+random.uniform(-1.0, 1.0) for human in living],
                           'radius': [self.person_radius for human in living],
                           'name': [human.name for human in living],
                           'type': ["child" if human.mobility_planner.follows_adult_schedule else "adult" for human in living]})


        joint_set = self.building_df.append(people_set)
        try:
            self.fig = px.scatter(joint_set, x="x", y="y", size="radius", hover_name="name", color="type",
                                  color_discrete_map=plot_pallet)
        except ValueError:
            self.fig = px.scatter(joint_set, x="x", y="y", size="radius", hover_name="name", color="type",
                                  color_discrete_map=plot_pallet)

        self.fig.update_layout(dragmode="pan")
        #self.fig = px.scatter(joint_set, x="x", y="y", color="colour", size="radius", hover_name="name")
        #self.ax.scatter(joint_set.x, joint_set.y, s=joint_set.radius, c=joint_set.colour)

plot_pallet = {"hospital": "red",
                "park": "yellow",
                "store": "orange",
                "school": "brown",
                "workplace": "purple",
                "senior_residence": "pink",
                "home": "green",
                "misc": "maroon",
                "adult": "blue",
                "child": "teal"}