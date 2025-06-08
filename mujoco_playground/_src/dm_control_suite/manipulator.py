# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Manipulator environment."""

from typing import Any, Dict, Optional, Union

from lxml import etree
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np  

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

_CLOSE = .01 
_P_IN_HAND = .1  # Probabillity of object-in-hand initial state
_P_IN_TARGET = .1  # Probabillity of object-in-target initial state
_ARM_JOINTS = ['arm_root', 'arm_shoulder', 'arm_elbow', 'arm_wrist',
               'finger', 'fingertip', 'thumb', 'thumbtip']
_ALL_PROPS = frozenset(['ball', 'target_ball', 'cup',
                        'peg', 'target_peg', 'slot'])
_TOUCH_SENSORS = ['palm_touch', 'finger_touch', 'thumb_touch',
                  'fingertip_touch', 'thumbtip_touch']
_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "manipulator.xml"

def default_config()-> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.001,
      episode_length=1000,
      action_repeat=1,
      vision=False,
  )

def make_model(xml, use_peg, insert):
  """Returns the model XML string."""
  parser = etree.XMLParser(remove_blank_text=True)
  mjcf = etree.XML(xml, parser)

  # Select the desired prop.
  if use_peg:
    required_props = ['peg', 'target_peg']
    if insert:
      required_props += ['slot']
  else:
    required_props = ['ball', 'target_ball']
    if insert:
      required_props += ['cup']

  # Remove unused props
  for unused_prop in _ALL_PROPS.difference(required_props):
    prop = mjcf.find('.//{}[@name={!r}]'.format('body', unused_prop))
    prop.getparent().remove(prop)

  return etree.tostring(mjcf, pretty_print=True)

@jax.jit
def print_jax(x):
    jax.debug.print("{x}", x=x)

class Bring(mjx_env.MjxEnv):
    def __init__(
        self,
        sparse: bool = False,
        use_peg: bool = False,
        insert: bool = False, 
        fully_observable: bool = True,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config=config, config_overrides=config_overrides)

        self._xml_path = _XML_PATH.as_posix()
        self._model_assets = common.get_assets()
        self._xml = make_model(_XML_PATH.read_text(), use_peg, insert)

        self._mj_model = mujoco.MjModel.from_xml_string(
            self._xml, self._model_assets
        )
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)

        self._use_peg = use_peg
        self._target = 'target_peg' if  use_peg else 'target_ball'
        self._receptacle = 'slot' if self._use_peg else 'cup'
        self._goal = self._receptacle if insert else self._target
        self._object = 'peg' if self._use_peg else 'ball'
        self._object_joints = ['_'.join([self._object, dim]) for dim in 'xzy']
        self._insert = insert
        self._fully_observable = fully_observable

        self._post_init()

    def _post_init(self) -> None:
        arm_jids = []
        for joint in _ARM_JOINTS:
            arm_jids.append(self._mj_model.joint(joint).id)
        self._arm_qposadrs = self._mj_model.jnt_qposadr[arm_jids]
        # print(self._arm_qposadrs)
        self._arm_lowers = self._mj_model.jnt_range[arm_jids, 0]
        self._arm_uppers = self._mj_model.jnt_range[arm_jids, 1]

        self.sensor_jids = []
        for sensor in _TOUCH_SENSORS:
            self.sensor_jids.append(self._mj_model.sensor(sensor).id)
        self.sensor_jids = jp.array(self.sensor_jids)

        self._object_ids = []
        for joint in [self._object + '_x', self._object + '_y', self._object + '_z']:
            self._object_ids.append(self._mj_model.joint(joint).id)
        self._object_qposadr = self._mj_model.jnt_qposadr[self._object_ids]

        self._target_jid = self._mj_model.body(self._object).id
        
        self._hand_jid = self._mj_model.body('hand').id

        self._grasp_id = self._mj_model.site('grasp').id

        self._get_reward = self._peg_reward if self._use_peg else self._ball_reward
    
    def reset(self, rng: jax.Array) -> mjx_env.State:
        
        data = mjx_env.init(self.mjx_model)
        rng1, rng2, rng3 = jax.random.split(rng, 3)

        qpos = jp.zeros(self.mjx_model.nq)
        qpos = qpos.at[self._arm_qposadrs].set(
            jax.random.uniform(
                rng1, (len(self._arm_qposadrs),), minval=self._arm_lowers, maxval=self._arm_uppers,
            )
        )
        qpos = qpos.at[self._arm_qposadrs[4]].set(qpos[self._arm_qposadrs[6]])
        
        qvel = 0.01 * jax.random.normal(rng2, (self.mjx_model.nv,))

        target_x = jax.random.uniform(rng2, (), minval=-.4, maxval=.4)
        target_z = jax.random.uniform(rng2, (), minval=.1, maxval=.4)

        if self._insert:
           target_angle = jax.random.uniform(rng3, (), minval=-jp.pi/3, maxval=jp.pi/3)
        else:
           target_angle = jax.random.uniform(rng3, (), minval=-jp.pi, maxval=jp.pi)
        
        xpos = data.xpos.copy()
        xquat = data.xquat.copy()
        site_xpos = data.site_xmat.copy()
        site_xmat = data.site_xmat.copy()
        
        xpos = xpos.at[self._target_jid, 0].set(target_x)
        xpos = xpos.at[self._target_jid, 2].set(target_z)
        xquat = xquat.at[self._target_jid, 0].set(jp.cos(target_angle/2))
        xquat = xquat.at[self._target_jid, 2].set(jp.sin(target_angle/2))

        object_init_probs = np.array([_P_IN_HAND, _P_IN_TARGET, 1-_P_IN_HAND-_P_IN_TARGET])
        choice_options = np.array([0, 1, 2])
        init_type = np.random.choice(a=choice_options, p=object_init_probs)

        if init_type == 1:
            object_x = target_x
            object_z = target_z
            object_angle = target_angle
        elif init_type == 0:
            object_x = site_xpos[self._grasp_id, 0]
            object_z = site_xpos[self._grasp_id, 2]
            grasp_direction = site_xmat[self._grasp_id, [0, 6]] # xx zx
            object_angle = jp.pi-jp.arctan2(grasp_direction[1], grasp_direction[0])
        else:
            object_x = jax.random.uniform(rng3, minval=-.5, maxval=.5)
            object_z = jax.random.uniform(rng3, minval=0, maxval=.7)
            object_angle = jax.random.uniform(rng3, minval=0, maxval=2*jp.pi)
            qvel = qvel.at[self._object_qposadr[0]].set(jax.random.uniform(rng3, minval=-5, maxval=5))
        
        qpos = qpos.at[self._object_qposadr].set(jp.array([object_x, object_z, object_angle]))
        data = data.replace(qpos=qpos, qvel=qvel, xpos=xpos, xquat=xquat)

        metrics = {
            "reward/ball_target_prox": jp.zeros(()),
            "reward/grasp_peg": jp.zeros(()),
            "reward/pinch_peg": jp.zeros(()),
            "reward/grasping": jp.zeros(()),
            "reward/bring_peg": jp.zeros(()),
            "reward/bring_tip": jp.zeros(()),
            "reward/bringing": jp.zeros(()),
        }

        info = {"rng": rng}
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
        reward = self._get_reward(data, action, state.info, state.metrics)  # pylint: disable=redefined-outer-name
        obs = self._get_obs(data, state.info)
        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)
        state = mjx_env.State(data, obs, reward, done, state.metrics, state.info)
        return state

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        del info  # Unused.
        obs = jp.concatenate([
           self.bounded_joint_pos(data, self._arm_qposadrs).flatten(),
           self.joint_vel(data, self._arm_qposadrs),
           self.touch(data),
        ])
        if self._fully_observable:
            return jp.concatenate([
                obs,
                self.body_2d_pose(data, self._hand_jid),
                self.body_2d_pose(data, self._object_ids),
                self.joint_vel(data, self._object_qposadr),
                self.body_2d_pose(data, self._target_jid),
            ])
        return obs
    
    def bounded_joint_pos(self, data, joint_adrs):
        """Returns joint positions as (sin, cos) values."""
        joint_pos = data.qpos[joint_adrs]
        return jp.vstack([jp.sin(joint_pos), jp.cos(joint_pos)]).T

    def joint_vel(self, data, joint_adrs):
        """Returns joint velocities."""
        return data.qvel[joint_adrs]
    
    def touch(self, data):
        sensor_data = data.sensordata[self.sensor_jids]
        return jp.log1p(sensor_data)

    def body_2d_pose(self, data, body_ids, orientation=True):
        """Returns positions and/or orientations of bodies."""
        pos_x = data.xpos[body_ids, 0]
        pos_z = data.xpos[body_ids, 1]
        if orientation:
            ori_qw = data.xquat[body_ids, 0]
            ori_qy = data.xquat[body_ids, 2]
            return jp.hstack([pos_x, pos_z, ori_qw, ori_qy])
        else:
            return jp.hstack([pos_x, pos_z])

    def site_distance(self, data, site1, site2):
        site1_id = self._mj_model.site(site1).id
        site2_id = self._mj_model.site(site2).id
        site_positions = jp.stack([data.site_xpos[site1_id], data.site_xpos[site2_id]], axis=0)
        site1_to_site2 = jp.diff(site_positions, axis=0)
        return jp.linalg.norm(site1_to_site2)
    
    def _is_close(self, distance):
        return reward.tolerance(distance, (0, _CLOSE), _CLOSE*2)

    def _ball_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
    ) -> jax.Array:
        return self._is_close(self.site_distance(data, 'ball', 'target_ball'))
    
    def _peg_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
    ) -> jax.Array:
        grasp = self._is_close(self.site_distance(data, 'peg_grasp', 'grasp'))
        pinch = self._is_close(self.site_distance(data, 'peg_pinch', 'pinch'))
        grasping = (grasp + pinch) / 2
        bring = self._is_close(self.site_distance(data, 'peg', 'target_peg'))
        bring_tip = self._is_close(self.site_distance(data, 'target_peg_tip',
                                                        'peg_tip'))
        bringing = (bring + bring_tip) / 2
        return max(bringing, grasping/3)

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model