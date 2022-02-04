from flask import Blueprint, request, jsonify, Response
from covid19sim.interactivity.controlled_run import ControlledSimulation

import InterSim.api.library.global_vars as gv

"""
This file defines the launchpage through Blueprints, and implements any needed background logic.
"""
sim = Blueprint('sim', __name__)

class _CityRepresentation:
    def __init__(self):
        self.activities = gv.CurrentSim.ACTIVITIES
        self.people = _serialize_human_definitions()
        self.buildings = {
            #'covid_testing_facility': gv.CurrentSim.city.covid_testing_facility,
            'hospitals': _serialize_hospitals(),
            'parks': _serialize_locations(gv.CurrentSim.city.parks),
            'schools': _serialize_locations(gv.CurrentSim.city.schools),
            'senior_residences': _serialize_locations(gv.CurrentSim.city.senior_residences),
            'stores': _serialize_locations(gv.CurrentSim.city.stores),
            'miscs': _serialize_locations(gv.CurrentSim.city.miscs),
            'workplaces': _serialize_locations(gv.CurrentSim.city.workplaces),
            'households': _serialize_locations(gv.CurrentSim.city.households)
        }

class _StatusUpdate:
    def __init__(self):
        self.people = _serialize_reduced_human()

def _serialize_hospitals():
    hospitals = [{'hospital':{
                        'name': h.name,
                        'position':{
                            'lat': h.lat,
                            'lon': h.lon},
                        'area': h.area,
                        'capacity': h.bed_capacity,
                        'bed_occupancy': h.bed_occupany,
                        'patients': [p for p in h.patients],
                        'icu': {
                            'position':{
                                'lat': h.icu.lat,
                                'lon': h.icu.lon},
                            'capacity': h.icu.bed_capacity,
                            'bed_occupancy': h.icu.bed_occupany,
                            'patients': [p for p in h.icu.patients]}
                        }
                } for h in gv.CurrentSim.city.hospitals]
    return hospitals


def _serialize_locations(location_list):
    location = [{'location':{
                'name': l.name,
                'position': {
                    'lat': l.lat,
                    'lon': l.lon},
                'area': l.area,
                'open': l.is_open_for_business},
                'open_days': l.open_days,
                'open_time': l.opening_time,
                'close_time': l.closing_time
            } for l in location_list]
    return location

def _serialize_human_definitions():
    people = [{'person':{
        'name': p.name,
        'sex': p.sex,
        'age': p.age,
        'position': {
            'lat': p.lat,
            'lon': p.lon},
        'obs_position': {
            'lat': p.obs_lat,
            'lon': p.obs_lon},
        'household': p.household.name if p.household is not None else None,
        'workplace': p.workplace.name if p.workplace is not None else None,
        'location': p.location.name if p.location is not None else None,
        'arrival_time': p.location_start_time,
        'departure_time': p.location_leaving_time,
        'schedule': _serialize_schedule(p.mobility_planner.schedule_for_day),

        'is_dead': p.is_dead,

        'allergies': p.has_allergy_symptoms,
        'cold_start': p.has_cold,
        'flu_start': p.has_flu,
        'covid_start': p.has_covid,
        'infected_contacts': p.n_infectious_contacts,
    }} for p in gv.CurrentSim.city.humans]
    return people


def _serialize_reduced_human():
    people = [{'person': {
        'name': p.name,
        'position': {
            'lat': p.lat,
            'lon': p.lon},
        'obs_position': {
            'lat': p.obs_lat,
            'lon': p.obs_lon},
        'workplace': p.workplace.name if p.workplace is not None else None,
        'schedule': _serialize_schedule(p.mobility_planner.schedule_for_day),
        'is_dead': p.is_dead,
        'allergies': p.has_allergy_symptoms,
        'cold_start': p.has_cold,
        'flu_start': p.has_flu,
        'covid_start': p.has_covid,
        'infected_contacts': p.n_infectious_contacts,
    }} for p in gv.CurrentSim.city.humans]
    return people


def _serialize_schedule(schedule):
    sched = [{
        'name': a.name,
        'location': a.location.name,
        'duration': a.duration,
        'start_time': a.start_time,
        'end_time': a.end_time,
        'human_dies': a.human_dies,
    }for a in list(schedule)]
    return sched

def _get_full_city_representation(status: int = 200):
    city = vars(_CityRepresentation())
    resp = jsonify({
        'response': city,
        'status': 200,
        'mimetype': 'application/json'})
    return resp

def _get_update_representation(status: int = 200):
    people = vars(_StatusUpdate())
    resp = jsonify({
        'response': people,
        'status': 200,
        'mimetype': 'application/json'})
    return resp

@sim.route('/', methods=['GET'])
def get_sim_state():
    if gv.CurrentSim is None:
        return Response('Failure: No active sim', status=500)
    return _get_full_city_representation(200)

@sim.route('/init', methods=['POST', 'PUT'])
def init_sim():
    if gv.CurrentConfig is None:
        return Response('Failure: No active config', status=500)
    gv.CurrentSim = ControlledSimulation(gv.CurrentConfig)
    return _get_full_city_representation(200)

@sim.route('/step', methods=['POST', 'PUT'])
def step_sim():
    ts = gv.CurrentSim.env.ts_now+1
    gv.CurrentSim.step(ts)
    if ts == gv.CurrentSim.end_time:
        gv.CurrentSim.end_sim()
        return Response('All Done', status=204)
    return _get_update_representation(200)

@sim.route('/update', methods=['PUT'])
def update_sim():
    pass

@sim.route('/teststep', methods=['get'])
def test_step():
    ts = gv.CurrentSim.env.ts_now+1
    gv.CurrentSim.step(ts)
    return Response('OK', status=200)