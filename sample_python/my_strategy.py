import model
from model.entity_type import EntityType as et
import pandas as pd
# from numba import njit
from sklearn.cluster import KMeans
from math import hypot
from math import sqrt
import random
import numpy as np

import datetime  # DEBUG
#import pickle  # DEBUG


class MyStrategy:

    def __init__(self):
        self.debug = False
        self.inicialization = True
        self.house_builders_batch_size = 2
        self.silent_period = 3
        self.repairers_per_target = 3
        self.house_side_switch = 150
        self.max_builder_distance = 0
        self.population_use = 0
        self.population_provide = 0
        self.ranged_base_count = 0
        self.enemy_centroids = pd.DataFrame([[6,  21], [ 18, 18], [21, 6]], columns=['x', 'y'])
        self.ban_houses = pd.DataFrame([[6, 6], [6, 16], [16, 6], [16, 16]], columns=['nx', 'ny'])
        self.complete_x = {}
        self.complete_y = {}
        self.busy_builders = []
        self.resource_wall_x = {}
        self.resource_wall_y = {}
        self.time_report = {
            'tick': []
        }

    def init_entities(self, timer_start, timer_end, player_view):
        if self.debug:
            timer_start['init'] = datetime.datetime.now()
        entity_storage = []
        self.population_use = 0
        self.population_provide = 0
        prev_ranged_base_count = self.ranged_base_count
        self.ranged_base_count = 0
        self.busy_builders = []
        self.resource_wall_x = [0 for _ in range(0, player_view.map_size)]
        self.resource_wall_y = [0 for _ in range(0, player_view.map_size)]
        for entity in player_view.entities:
            if entity.entity_type == et.RESOURCE:
                if entity.position.x < 3:
                    self.resource_wall_y[entity.position.y] += 1
                if entity.position.y < 3:
                    self.resource_wall_x[entity.position.x] += 1
            else:
                e_type = player_view.entity_properties[entity.entity_type]
                entity_storage.append([
                    entity.player_id,
                    entity.id,
                    entity.entity_type,
                    entity.position.x,
                    entity.position.y,
                    hypot(entity.position.x, entity.position.y),
                    e_type.max_health-entity.health
                ])
                if entity.player_id == player_view.my_id:
                    if entity.entity_type == et.RANGED_BASE:
                        self.ranged_base_count += 1
                    if entity.entity_type == et.HOUSE:
                        if entity.position.y == 0:
                            self.complete_x[entity.position.x] = True
                        elif entity.position.x == 0:
                            self.complete_y[entity.position.y] = True
                    self.population_use += e_type.population_use
                    self.population_provide += e_type.population_provide

        #print('pop',self.population_use, self.population_provide)
        df = pd.DataFrame(entity_storage, columns=[
            'player_id',
            'entity_id',
            'entity_type',
            'x',
            'y',
            'zero_distance',
            'health_gap'
        ])
        if self.ranged_base_count > prev_ranged_base_count:
            self.inicialization = True

        if self.debug:
            timer_end['init'] = datetime.datetime.now()

        return df

    def enemy_clusters(self, timer_start, timer_end, my_id, tick, entities):
        if self.debug:
            timer_start['enemy_clusters'] = datetime.datetime.now()

        # get max builder distance
        mask = (entities.player_id == my_id) & (entities.entity_type == et.BUILDER_UNIT)
        builders = pd.DataFrame(entities[mask], columns = ['x', 'y', 'zero_distance'])
        builders.sort_values(['zero_distance'], ascending = False, inplace = True)
        if len(builders):
            self.max_builder_distance = int(builders['zero_distance'].head(1).values)
        else:
            self.max_builder_distance = 25

        mask = (entities.player_id != my_id) & \
               (entities.zero_distance < self.max_builder_distance + 5)
        enemy = entities[mask]
        if len(enemy):
            enemy = pd.DataFrame(enemy, columns=['x', 'y'])
            kmeans = KMeans(n_clusters=min([len(enemy),3])).fit(enemy)
            self.enemy_centroids = pd.DataFrame(kmeans.cluster_centers_.astype(int), columns = ['x','y'])

        if self.debug:
            timer_end['enemy_clusters'] = datetime.datetime.now()

    def attack_action_v2(self, timer_start, timer_end, my_id, sight_range, entities, actions):
        if self.debug:
            timer_start['attack_action'] = datetime.datetime.now()

        # get attackers
        mask = (entities['player_id'] == my_id) & \
               (
                       (entities['entity_type'] == et.MELEE_UNIT) |
                       (entities['entity_type'] == et.RANGED_UNIT)
               )
        units = pd.DataFrame(entities[mask], columns=['entity_id', 'x', 'y'])
        units.columns = ['entity_id', 'ux', 'uy']
        i = 0
        for cx, cy in self.enemy_centroids.values:
            units['d' + str(i)] = np.hypot(units.ux - cx, units.uy - cy)
            i += 1

        busy = []
        unit_len = len(units)
        counter_len = len(self.enemy_centroids)
        unit_counter = 0
        current_centroid = 0
        units.sort_values(['d' + str(current_centroid)], ascending=True, inplace=True)
        for ex, ey in self.enemy_centroids.values:
            target = model.vec2_int.Vec2Int(ex, ey)

            m, b, a, r = None, None, None, None
            m = model.move_action.MoveAction(target, True, True)
            a = model.attack_action.AttackAction(
                target=None,
                auto_attack=model.auto_attack.AutoAttack(
                    pathfind_range=sight_range,
                    valid_targets=[]
                )
            )
            for entity_id in units.entity_id:
                if not entity_id in busy:
                    actions[entity_id] = model.EntityAction(m, b, a, r)
                    busy.append(entity_id)
                    unit_counter += 1
                    if unit_counter >= unit_len / counter_len:
                        unit_counter = 0
                        current_centroid += 1
                        units.sort_values(['d' + str(current_centroid - 1)], ascending=True, inplace=True)
                        break

        if self.debug:
            timer_end['attack_action'] = datetime.datetime.now()

        return actions

    def init_buildings(self, timer_start, timer_end, player_view, entities, actions, tick):
        if self.debug:
            timer_start['init_buildings'] = datetime.datetime.now()

        if self.inicialization:
            # factory
            mask = (entities['player_id'] == player_view.my_id) & \
                   (
                           (entities['entity_type'] == et.BUILDER_BASE) |
                           (entities['entity_type'] == et.MELEE_BASE) |
                           (entities['entity_type'] == et.RANGED_BASE)
                   )
            units = pd.DataFrame(entities[mask], columns=['entity_id', 'entity_type', 'x', 'y'])
            m, b, a, r = None, None, None, None
            for entity_id, entity_type, x, y in units.values:
                properties = player_view.entity_properties[entity_type]
                build_properties = properties.build
                e_type = build_properties.options[0]
                b = model.build_action.BuildAction(
                    e_type,
                    model.vec2_int.Vec2Int(
                        x + properties.size,
                        y + properties.size - 1,
                    )
                )
                actions[entity_id] = model.EntityAction(m, b, a, r)

            # turret
            mask = (entities['entity_type'] == et.TURRET) & (entities['player_id'] == player_view.my_id)
            units = pd.DataFrame(entities[mask], columns=['entity_id'])
            if len(units):
                m, b, a, r = None, None, None, None
                properties = player_view.entity_properties[et.TURRET]
                a = model.attack_action.AttackAction(
                    target=None,
                    auto_attack=model.auto_attack.AutoAttack(
                        pathfind_range=properties.sight_range,
                        valid_targets=[]
                    )
                )
                for entity_id in units['entity_id']:
                    actions[entity_id] = model.EntityAction(m, b, a, r)

            self.inicialization = False

        if self.debug:
            timer_end['init_buildings'] = datetime.datetime.now()

        return actions

    def reclaim_action(self, timer_start, timer_end, my_id, sight_range, map_size, entities, actions):
        if self.debug:
            timer_start['reclaim_action'] = datetime.datetime.now()

        mask = (entities['entity_type'] == et.BUILDER_UNIT) & \
               (entities['player_id'] == my_id)
        units = pd.DataFrame(entities[mask], columns=['entity_id'])
        if len(units):
            m, b, a, r = None, None, None, None
            target = model.vec2_int.Vec2Int(map_size - 1, map_size - 1)
            m = model.move_action.MoveAction(target, True, True)
            a = model.attack_action.AttackAction(
                target=None,
                auto_attack=model.auto_attack.AutoAttack(
                    pathfind_range=sight_range,
                    valid_targets=[et.RESOURCE]
                )
            )
            for entity_id in units['entity_id']:
                actions[entity_id] = model.EntityAction(m, b, a, r)

        if self.debug:
            timer_end['reclaim_action'] = datetime.datetime.now()

        return actions

    def repairers_action(self, timer_start, timer_end, my_id, entities, actions):
        if self.debug:
            timer_start['repairers_action'] = datetime.datetime.now()

        mask = (entities['player_id'] == my_id) & ( \
                    (entities['entity_type'] == et.BUILDER_BASE) | \
                    (entities['entity_type'] == et.MELEE_BASE) | \
                    (entities['entity_type'] == et.RANGED_BASE) | \
                    (entities['entity_type'] == et.TURRET) | \
                    (entities['entity_type'] == et.HOUSE)) & \
               (entities['health_gap'] > 0)
        damaged = pd.DataFrame(entities[mask], columns=['entity_id', 'x', 'y', 'health_gap'])
        damaged.columns = ['entity_id', 'dx', 'dy', 'health_gap']
        damaged.sort_values(['health_gap'], ascending=False, inplace=True)
        if len(damaged):
            busy = []
            mask = (entities['player_id'] == my_id) & (entities['entity_type'] == et.BUILDER_UNIT)
            for building_id, dx, dy, health_gap in damaged.values:
                repairers = pd.DataFrame(entities[mask], columns=['entity_id', 'x', 'y'])
                repairers.columns = ['entity_id', 'rx', 'ry']
                repairers['distance'] = np.hypot(repairers.rx - dx, repairers.ry - dy)
                repairers.sort_values(['distance'], ascending=True, inplace=True)
                repairers_count = 0
                for repairer_id, rx, ry, distance in repairers.values:
                    repairer_id = int(repairer_id)
                    if not repairer_id in busy:
                        m, b, a, r = None, None, None, None
                        target = model.vec2_int.Vec2Int(dx, dy)
                        r = model.repair_action.RepairAction(target=building_id)
                        m = model.move_action.MoveAction(target, True, True)
                        actions[repairer_id] = model.EntityAction(m, b, a, r)
                        busy.append(repairer_id)
                        repairers_count += 1
                        if repairers_count >= self.repairers_per_target:
                            break

        if self.debug:
            timer_end['repairers_action'] = datetime.datetime.now()

        return actions

    def house_place_available_x(self,x):
        return self.resource_wall_x[x] + self.resource_wall_x[x + 1] + self.resource_wall_x[x + 2] == 0

    def house_place_available_y(self,y):
        return self.resource_wall_y[y] + self.resource_wall_y[y + 1] + self.resource_wall_y[y + 2] == 0

    def house_builders_action_v2(self, timer_start, timer_end, my_id, entities, actions, initial_cost, resource, map_size):
        if self.debug:
            timer_start['house_builders_action_v2'] = datetime.datetime.now()

        if self.population_provide - self.population_use < 10 and initial_cost < resource:
            hy = 0
            builders_of_this_batch_count = 0
            for hx in range(0,map_size-3,3):
                if not hx in self.complete_x and self.house_place_available_x(hx):
                    mask = (entities['player_id'] == my_id) & \
                           (entities['entity_type'] == et.BUILDER_UNIT)# & \
                           #(entities['zero_distance'] < max(30,len(self.complete_x)*3+1))
                    builders = pd.DataFrame(entities[mask], columns=['entity_id', 'x', 'y'])
                    builders['distance'] = np.hypot(builders.x - hx + 3, builders.y - hy + 3)
                    builders.sort_values(['distance'], ascending=True, inplace=True)
                    #x_builder = pd.DataFrame(builders.head(10))
                    x_builders = pd.DataFrame(builders)
                    for builder_id, bx, by, distance in x_builders.values:
                        builder_id = int(builder_id)
                        if not builder_id in self.busy_builders:
                            m, b, a, r = None, None, None, None
                            target = model.vec2_int.Vec2Int(hx, hy)
                            b = model.build_action.BuildAction(et.HOUSE, model.vec2_int.Vec2Int(hx, hy))
                            m = model.move_action.MoveAction(model.vec2_int.Vec2Int(hx + 2, hy + 3), True, True)
                            actions[builder_id] = model.EntityAction(m, b, a, r)
                            """print(builder_id,
                                  'x',
                                  hx,
                                  self.resource_wall_x[hx],
                                  self.resource_wall_x[hx + 1],
                                  self.resource_wall_x[hx + 2],
                                  len(x_builders))"""
                            self.busy_builders.append(builder_id)
                            builders_of_this_batch_count += 1
                            break
                    if builders_of_this_batch_count > self.house_builders_batch_size:
                        break

        if self.population_provide - self.population_use < 10 and initial_cost < resource:
            hx = 0
            builders_of_this_batch_count = 0
            for hy in range(6,map_size-3,3):
                if not hy in self.complete_y and self.house_place_available_y(hy):
                    mask = (entities['player_id'] == my_id) & \
                           (entities['entity_type'] == et.BUILDER_UNIT)# & \
                           #(entities['zero_distance'] < max(30,len(self.complete_y)*3+1))
                    builders = pd.DataFrame(entities[mask], columns=['entity_id', 'x', 'y'])
                    builders['distance'] = np.hypot(builders.x - hx + 3, builders.y - hy + 3)
                    builders.sort_values(['distance'], ascending=True, inplace=True)
                    #y_builder = pd.DataFrame(builders.head(1))
                    y_builders = pd.DataFrame(builders)
                    for builder_id, bx, by, distance in y_builders.values:
                        builder_id = int(builder_id)
                        if not builder_id in self.busy_builders:
                            m, b, a, r = None, None, None, None
                            target = model.vec2_int.Vec2Int(hx, hy)
                            b = model.build_action.BuildAction(et.HOUSE, model.vec2_int.Vec2Int(hx, hy))
                            m = model.move_action.MoveAction(model.vec2_int.Vec2Int(hx + 3, hy + 2), True, True)
                            actions[builder_id] = model.EntityAction(m, b, a, r)
                            self.busy_builders.append(builder_id)
                            builders_of_this_batch_count += 1
                            break
                    if builders_of_this_batch_count > self.house_builders_batch_size:
                        break

        if self.debug:
            timer_end['house_builders_action_v2'] = datetime.datetime.now()

        return actions

    def ranged_base_builders_action(self, timer_start, timer_end, my_id, entities, actions, initial_cost, resource):
        if self.debug:
            timer_start['ranged_base_builders_action'] = datetime.datetime.now()

        if initial_cost < resource and self.ranged_base_count < 2:

            hx = 22
            hy = 5
            mask = (entities['player_id'] == my_id) & (entities['entity_type'] == et.BUILDER_UNIT)
            builders = pd.DataFrame(entities[mask], columns=['entity_id', 'x', 'y'])
            builders['distance'] = np.hypot(builders.x - hx - 1, builders.y - hy)
            builders.sort_values(['distance'], ascending=True, inplace=True)
            x_builders = pd.DataFrame(builders)
            for builder_id, bx, by, distance in x_builders.values:
                builder_id = int(builder_id)
                if not builder_id in self.busy_builders:
                    m, b, a, r = None, None, None, None
                    target = model.vec2_int.Vec2Int(hx, hy)
                    b = model.build_action.BuildAction(et.RANGED_BASE, model.vec2_int.Vec2Int(hx, hy))
                    m = model.move_action.MoveAction(model.vec2_int.Vec2Int(hx + 2, hy + 3), True, True)
                    actions[builder_id] = model.EntityAction(m, b, a, r)
                    self.busy_builders.append(builder_id)
                    break

        if self.debug:
            timer_end['ranged_base_builders_action'] = datetime.datetime.now()

        return actions

    def get_action(self, player_view, debug_interface):
        timer_start = {}
        timer_end = {}
        actions = {}

        if player_view.current_tick % self.silent_period == 0:
            my_id = player_view.my_id
            # parse player_view
            entities = self.init_entities(timer_start, timer_end, player_view)
            # only first time init buildings
            self.init_buildings(timer_start, timer_end, player_view, entities, actions, player_view.current_tick)
            # define near enemy clusters
            self.enemy_clusters(timer_start, timer_end, my_id, player_view.current_tick, entities)
            # attack actions
            properties = player_view.entity_properties[et.MELEE_UNIT]
            sight_range= properties.sight_range
            actions = self.attack_action_v2(timer_start, timer_end, my_id, sight_range, entities, actions)
            # reclaim
            properties = player_view.entity_properties[et.BUILDER_UNIT]
            sight_range = properties.sight_range
            actions = self.reclaim_action(
                timer_start,
                timer_end,
                my_id,
                sight_range,
                player_view.map_size,
                entities,
                actions
            )
            # build houses
            actions = self.house_builders_action_v2(
                timer_start,
                timer_end,
                my_id,
                entities,
                actions,
                player_view.entity_properties[et.HOUSE].initial_cost,
                player_view.players[my_id-1].resource,
                player_view.map_size
                #player_view.current_tick
            )
            # build ranged base
            actions = self.ranged_base_builders_action(
                timer_start,
                timer_end,
                my_id,
                entities,
                actions,
                player_view.entity_properties[et.RANGED_BASE].initial_cost,
                player_view.players[my_id - 1].resource
            )
            # repair actions
            actions = self.repairers_action(timer_start, timer_end, my_id, entities, actions)

        # debug area
        if self.debug and player_view.current_tick % self.silent_period == 0:
            self.time_report['tick'].append(player_view.current_tick)
            for key in timer_start.keys():
                if not key in self.time_report.keys():
                    self.time_report[key] = []
                self.time_report[key].append((timer_end[key] - timer_start[key]).microseconds / 1000)

        """if self.debug and player_view.current_tick == 500:
            pickle.dump(self.time_report, file=open('time_report_' + str(player_view.my_id) + '.dat', 'wb'))
            print('log saved')"""

        return model.Action(actions)

    def debug_update(self, player_view, debug_interface):
        debug_interface.send(model.DebugCommand.Clear())
        debug_interface.get_state()