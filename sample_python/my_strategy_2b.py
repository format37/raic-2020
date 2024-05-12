import model
from model.entity_type import EntityType as et
import pandas as pd
# from numba import njit

import datetime  # DEBUG
import pickle  # DEBUG


class MyStrategy:

    def __init__(self):
        self.debug = True
        self.inicialization = True
        self.free_builders = 2
        self.planned_house_builders = 1
        self.planned_repairers = 6
        self.damaged = pd.DataFrame()
        self.time_report = {
            'tick': [],
            'my_units': [],
            'all_units': []
        }
        self.tasks = pd.DataFrame(columns=[
            'entity_id',
            'entity_type',
            'task'
        ])
        self.simcity = pd.DataFrame(
            [
                [11, 5, et.HOUSE],
                [11, 9, et.HOUSE],
                [11, 13, et.HOUSE],
                [11, 17, et.HOUSE],
                [7, 11, et.HOUSE],
                [1, 11, et.HOUSE],
                [1, 7, et.HOUSE],
                [1, 1, et.HOUSE],
                [7, 1, et.HOUSE],
                [11, 1, et.HOUSE],
                [15, 1, et.HOUSE],
                [19, 1, et.HOUSE],
                [15, 11, et.HOUSE],
                [1, 15, et.HOUSE],
                [1, 19, et.HOUSE],
                [1, 25, et.HOUSE],
                [5, 21, et.HOUSE],
                [9, 21, et.HOUSE],
                [1, 29, et.HOUSE],
                [1, 33, et.HOUSE],
                [5, 25, et.HOUSE],
                [5, 29, et.HOUSE],
                [5, 33, et.HOUSE],
                [9, 25, et.HOUSE],
                [9, 29, et.HOUSE],
                [9, 33, et.HOUSE],
                [13, 25, et.HOUSE],
                [13, 29, et.HOUSE],
                [13, 33, et.HOUSE],
            ],
            columns=['x', 'y', 'entity_type'])

    def action_base(self, player_view, timer_start, timer_end, entities, actions):
        if self.debug:
            timer_start['action_base'] = datetime.datetime.now()

        if self.inicialization:
            # base
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

            self.inicialization = False

        if self.debug:
            timer_end['action_base'] = datetime.datetime.now()

        return actions

    def update_house_builders(self, player_view, timer_start, timer_end, entities):
        if self.debug:
            timer_start['update_house_builders'] = datetime.datetime.now()

        my_id = player_view.my_id
        properties = player_view.entity_properties[et.HOUSE]
        if properties.initial_cost * 2 < player_view.players[my_id-1].resource:
            mask = (self.tasks['entity_type'] == et.BUILDER_UNIT) & (self.tasks['task'] == 'house')
            house_builders = self.tasks[mask]
            mask = (self.tasks['entity_type'] == et.BUILDER_UNIT) & ( \
                        (self.tasks['task'] == 'repair') | \
                        (self.tasks['task'] == 'house'))
            busy_builders = self.tasks[mask]
            mask = (entities['entity_type'] == et.BUILDER_UNIT) & (entities['player_id'] == player_view.my_id)
            all_builders = entities[mask]
            free_builders = all_builders[~all_builders.entity_id.isin(busy_builders.entity_id)]
            if len(house_builders) < self.planned_house_builders and len(free_builders) > self.free_builders:
                # assign task to free builder
                new_workers = pd.DataFrame(all_builders.head(self.planned_house_builders - len(house_builders)))
                new_workers['task'] = 'house'
                self.tasks = pd.concat([self.tasks, new_workers], sort=False, axis=0)
        else:
            # drop build house tasks
            self.tasks = pd.DataFrame(self.tasks[self.tasks.task != 'house'])

        if self.debug:
            timer_end['update_house_builders'] = datetime.datetime.now()

    def update_repairer(self, player_view, timer_start, timer_end, entities):
        if self.debug:
            timer_start['update_repairer'] = datetime.datetime.now()

        # TODO: set repairer exactly for (damaged) entity id
        # get neares damaged entities
        mask = (entities['entity_type'] != et.RESOURCE) & \
               (entities['player_id'] == player_view.my_id) & \
               (entities['health'] < entities['max_health']) & \
               (entities['entity_type'] != et.RANGED_UNIT) & \
               (entities['entity_type'] != et.MELEE_UNIT) & \
               (entities['entity_type'] != et.BUILDER_UNIT)
        self.damaged = pd.DataFrame(entities[mask], columns=['entity_id', 'x', 'y'])
        if len(self.damaged) > 0:
            mask = (self.tasks['entity_type'] == et.BUILDER_UNIT) & (self.tasks['task'] == 'repair')
            repairers = self.tasks[mask]
            mask = (self.tasks['entity_type'] == et.BUILDER_UNIT) & ( \
                        (self.tasks['task'] == 'repair') | \
                        (self.tasks['task'] == 'house'))
            busy_builders = self.tasks[mask]
            mask = (entities['entity_type'] == et.BUILDER_UNIT) & (entities['player_id'] == player_view.my_id)
            all_builders = entities[mask]
            free_builders = all_builders[~all_builders.entity_id.isin(busy_builders.entity_id)]
            if len(repairers) < self.planned_repairers and len(free_builders) > self.free_builders:
                # assign task to free builder
                new_workers = pd.DataFrame(all_builders.head(self.planned_repairers - len(repairers)))
                new_workers['task'] = 'repair'
                self.tasks = pd.concat([self.tasks, new_workers], sort=False, axis=0)

        if self.debug:
            timer_end['update_repairer'] = datetime.datetime.now()

    def action_build_house(self, player_view, timer_start, timer_end, entities, actions):
        if self.debug:
            timer_start['action_build_house'] = datetime.datetime.now()

        # get builders
        builders = self.tasks[self.tasks.task == 'house']

        if len(builders) > 0:
            # get empty places
            mask = (entities['entity_type'] == et.HOUSE) & (entities['player_id'] == player_view.my_id)
            exists_houses = pd.DataFrame(entities[mask])
            exists_houses['id'] = (exists_houses['x'] + 100) * exists_houses['y']  # tokenize
            estimated_houses = pd.DataFrame(self.simcity[self.simcity.entity_type == et.HOUSE])
            estimated_houses['id'] = (estimated_houses['x'] + 100) * estimated_houses['y']  # tokenize
            # drop complete
            estimated_houses = pd.DataFrame(estimated_houses[~estimated_houses.id.isin(exists_houses.id)].head(1))
            if len(estimated_houses):
                # set builders action
                m, b, a, r = None, None, None, None
                bx = int(estimated_houses.x)
                by = int(estimated_houses.y)
                builders = pd.DataFrame(
                    entities[entities.entity_id.isin(builders.entity_id)],
                    columns=['entity_id', 'x', 'y']
                )
                for entity_id, ex, ey in builders.values:
                    if ex == bx - 1 and ey == by:
                        b = model.build_action.BuildAction(et.HOUSE, model.vec2_int.Vec2Int(bx, by))
                    else:
                        m = model.move_action.MoveAction(model.vec2_int.Vec2Int(bx - 1, by), True, True)
                    actions[entity_id] = model.EntityAction(m, b, a, r)

        if self.debug:
            timer_end['action_build_house'] = datetime.datetime.now()

        return actions

    def action_repair(self, player_view, timer_start, timer_end, entities, actions):
        if self.debug:
            timer_start['action_repair'] = datetime.datetime.now()

        # get neares damaged entities
        """
        mask = (entities['entity_type'] != et.RESOURCE) & \
               (entities['player_id'] == player_view.my_id ) & \
               (entities['health'] < entities['max_health']) & \
               (entities['entity_type'] != et.RANGED_UNIT) & \
               (entities['entity_type'] != et.MELEE_UNIT) & \
               (entities['entity_type'] != et.BUILDER_UNIT)
        damaged = pd.DataFrame(entities[mask], columns=['entity_id','x', 'y'])
        """
        if len(self.damaged)>0:
            self.damaged['xy'] = self.damaged['x'] + self.damaged['y']
            self.damaged.sort_values(['xy'], ascending = False, inplace = True)
            damaged = pd.DataFrame(self.damaged.head(1))
            # action
            for entity_id in self.tasks[self.tasks['task'] == 'repair'].entity_id.values:
                m, b, a, r = None, None, None, None
                x = damaged.x.values[0]
                y = damaged.y.values[0]
                target = model.vec2_int.Vec2Int(x, y)
                r = model.repair_action.RepairAction(target=damaged.entity_id.values[0])
                m = model.move_action.MoveAction(target, True, True)
                actions[entity_id] = model.EntityAction(m, b, a, r)
        else:
            # drop repair tasks
            self.tasks = pd.DataFrame(self.tasks[self.tasks.task != 'repair'])

        if self.debug:
            timer_end['action_repair'] = datetime.datetime.now()

        return actions

    def action_reclaim(self, player_view, timer_start, timer_end, entities, actions):
        if self.debug:
            timer_start['action_reclaim'] = datetime.datetime.now()

        mask = (entities['entity_type'] == et.BUILDER_UNIT) & (entities['player_id'] == player_view.my_id)
        units = pd.DataFrame(entities[mask], columns=['entity_id'])
        free_units = units[~units.entity_id.isin(self.tasks.entity_id)]
        units['task'] = 'reclaim'

        if len(free_units):
            # update tasks
            self.tasks = pd.concat([self.tasks, free_units], sort=False, axis=0)
            # action
            m, b, a, r = None, None, None, None
            properties = player_view.entity_properties[et.BUILDER_UNIT]
            target = model.vec2_int.Vec2Int(player_view.map_size - 1, player_view.map_size - 1)
            m = model.move_action.MoveAction(target, True, True)
            a = model.attack_action.AttackAction(
                target=None,
                auto_attack=model.auto_attack.AutoAttack(
                    pathfind_range=properties.sight_range,
                    valid_targets=[et.RESOURCE]
                )
            )
            for entity_id in free_units['entity_id']:
                actions[entity_id] = model.EntityAction(m, b, a, r)

        if self.debug:
            timer_end['action_reclaim'] = datetime.datetime.now()

        return actions

    def action_warrior(self, player_view, timer_start, timer_end, entities, actions):
        if self.debug:
            timer_start['action_warrior'] = datetime.datetime.now()

        mask = (entities['player_id'] == player_view.my_id) & \
               (
                       (entities['entity_type'] == et.MELEE_UNIT) |
                       (entities['entity_type'] == et.RANGED_UNIT)
               )
        units = pd.DataFrame(entities[mask], columns=['entity_id', 'entity_type'])
        free_units = pd.DataFrame(units[~units.entity_id.isin(self.tasks.entity_id)])
        free_units['task'] = 'attack'

        if len(free_units):
            # update tasks
            self.tasks = pd.concat([self.tasks, free_units], sort=False, axis=0)
            # action
            mask = (entities['player_id'] != player_view.my_id) & \
                   (
                           (entities['entity_type'] == et.MELEE_BASE) |
                           (entities['entity_type'] == et.RANGED_BASE) |
                           (entities['entity_type'] == et.TURRET) |
                           (entities['entity_type'] == et.HOUSE)
                   )
            selected_enemy = entities[mask].sort_values(['player_id'], ascending=False).head(1)
            if len(selected_enemy):
                ex = int(selected_enemy['x'])
                ey = int(selected_enemy['y'])
                m, b, a, r = None, None, None, None
                properties = player_view.entity_properties[et.MELEE_UNIT]
                target = model.vec2_int.Vec2Int(ex, ey)
                m = model.move_action.MoveAction(target, True, True)
                a = model.attack_action.AttackAction(
                    target=None,
                    auto_attack=model.auto_attack.AutoAttack(
                        pathfind_range=properties.sight_range,
                        valid_targets=[]
                    )
                )
                for entity_id in free_units['entity_id']:
                    actions[entity_id] = model.EntityAction(m, b, a, r)

        if self.debug:
            timer_end['action_warrior'] = datetime.datetime.now()

        return actions

    def action_turret(self, player_view, timer_start, timer_end, entities, actions):
        if self.debug:
            timer_start['action_turret'] = datetime.datetime.now()

        mask = (entities['entity_type'] == et.TURRET) & (entities['player_id'] == player_view.my_id)
        units = pd.DataFrame(entities[mask]['entity_id'], columns=['entity_id'])
        # action
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

        if self.debug:
            timer_end['action_turret'] = datetime.datetime.now()

        return actions

    def init_entities(self, player_view, timer_start, timer_end):
        if self.debug:
            timer_start['init'] = datetime.datetime.now()
        entity_storage = []
        for entity in player_view.entities:
            if not entity.entity_type == et.RESOURCE:
                entity_storage.append([
                    entity.player_id,
                    entity.id,
                    entity.entity_type,
                    entity.position.x,
                    entity.position.y,
                    entity.health,
                    player_view.entity_properties[entity.entity_type].max_health
                ])
        df = pd.DataFrame(entity_storage, columns=[
            'player_id',
            'entity_id',
            'entity_type',
            'x',
            'y',
            'health',
            'max_health'
        ])

        if self.debug:
            timer_end['init'] = datetime.datetime.now()

        return df

    def remove_died(self, timer_start, timer_end, entities):
        if self.debug:
            timer_start['remove_died'] = datetime.datetime.now()

            self.tasks = self.tasks[~self.tasks.entity_id.isin(entities.entity_id)]

        if self.debug:
            timer_end['remove_died'] = datetime.datetime.now()

    def get_action(self, player_view, debug_interface):
        timer_start = {}
        timer_end = {}
        if self.debug:
            timer_start['main'] = datetime.datetime.now()

        entities = self.init_entities(player_view, timer_start, timer_end)
        self.remove_died(timer_start, timer_end, entities)
        actions = {}
        actions = self.action_reclaim(player_view, timer_start, timer_end, entities, actions)
        actions = self.action_turret(player_view, timer_start, timer_end, entities, actions)
        actions = self.action_warrior(player_view, timer_start, timer_end, entities, actions)
        actions = self.action_base(player_view, timer_start, timer_end, entities, actions)
        self.update_house_builders(player_view, timer_start, timer_end, entities)
        self.action_build_house(player_view, timer_start, timer_end, entities, actions)
        self.update_repairer(player_view, timer_start, timer_end, entities)
        actions = self.action_repair(player_view, timer_start, timer_end, entities, actions)

        if self.debug:
            timer_end['main'] = datetime.datetime.now()
            self.time_report['tick'].append(player_view.current_tick)
            self.time_report['my_units'].append(len(entities[entities['player_id'] == player_view.my_id]) / 10)
            self.time_report['all_units'].append(len(entities) / 10)
            for key in timer_start.keys():
                if not key in self.time_report.keys():
                    self.time_report[key] = []
                self.time_report[key].append((timer_end[key] - timer_start[key]).microseconds / 1000)
            if player_view.current_tick == 990:
                pickle.dump(self.time_report, file=open('time_report_' + str(player_view.my_id) + '.dat', 'wb'))
                print('log saved')

        return model.Action(actions)

    def debug_update(self, player_view, debug_interface):
        debug_interface.send(model.DebugCommand.Clear())
        debug_interface.get_state()
