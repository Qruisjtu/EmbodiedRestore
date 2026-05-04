import numpy as np
from random import choice
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.models.objects import (
    BoxObject,
)
import robosuite.models.objects as objs
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat


class LiftTest(ManipulationEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        self.use_object_obs = use_object_obs

        self.placement_initializer = placement_initializer
        self.checkonce = 1
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        reward = 0.0

        if self._check_success():
            reward = 2.25
        elif self.reward_shaping:
            dist = self._gripper_to_target(
                gripper=self.robots[0].gripper, target=self.object.root_body, target_type="body", return_distance=True
            )
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=[self.object]):
                reward += 0.25
        
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _load_model(self):
        super()._load_model()

        # 机器人基座位置偏移
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # 创建桌子
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            is_randomize_material=True
        )
        self.arena_metadata:dict = mujoco_arena.arena_metadata()
        mujoco_arena.set_origin([0, 0, 0])
        
        
        milk = objs.MilkObject(name="milk")
        can = objs.CanObject(name="can")
        bottle = objs.BottleObject(name="bottle")
        lemon = objs.LemonObject(name="lemon")
        bread = objs.BreadObject(name = "bread")
        cereal = objs.CerealObject(name="cereal")
        square_nut= objs.SquareNutObject(name = "square_nut")
        round_nut = objs.RoundNutObject(name="round_nut")
        capsule = objs.CapsuleObject(name="capsule",size=(0.015, 0.015),rgba=(0.9,0.9,0.9,1))
        # pot = objs.pot_with_handles(name="pot")
        hammer = objs.HammerObject(name="hammer")
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        
        redcube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.object = choice([milk,can,redcube,bottle,lemon,bread,cereal,square_nut,round_nut,capsule,hammer]) #Train/Eval

        obj_rotation = 0
        tool = objs.RatchetingWrenchObject(name="wrench",handle_size=(0.06,0.01,0.005),height_1=0.005,height_2=0.005,outer_radius_1=0.02,outer_radius_2=0.02,inner_radius_1=0.01,inner_radius_2=0.01)
        bowl = objs.BowlObject(name='bowl')
        ketchup =  objs.KetchupObject(name="ketchup")
        juice = objs.JuiceObject(name="juice")
        soup = objs.SoupObject(name="soup")
        mug = objs.MugObject(name="mug")
        # self.object =choice([tool,ketchup,juice,mug,soup,bowl])
        if self.object is ketchup or self.object is juice or self.object is soup:
            obj_rotation = np.pi / 2

        # 物体随机放置采样器
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.object)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=[self.object],
                x_range=[-0.1, 0.1],
                y_range=[-0.2, 0.2],
                # rotation=obj_rotation,
                # rotation_axis='x',
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                # rng=self.rng
            )

        # 构建 ManipulationTask
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.object]  ,
        )

    def _setup_references(self):
        super()._setup_references()
        self.obj_body_id = self.sim.model.body_name2id(self.object.root_body)
        # self.can_body_id = self.sim.model.body_name2id(self.can.root_body)

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def obj_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.obj_body_id])

            @sensor(modality=modality)
            def obj_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.obj_body_id]), to="xyzw")

            sensors = [obj_pos, obj_quat]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            sensors += [
                self._get_obj_eef_sensor(full_pf, "obj_pos", f"{arm_pf}gripper_to_obj_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]

            names = [s.__name__ for s in sensors]

            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            
            for obj_pos, obj_quat, obj in object_placements.values():
                if obj.joints:
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                

    def _check_success(self):
        if self.checkonce :
            self.initheight = self.sim.data.body_xpos[self.obj_body_id][2]
        self.checkonce = 0
        obj_height = self.sim.data.body_xpos[self.obj_body_id][2]
        # table_height = self.model.mujoco_arena.table_offset[2]

        return obj_height > self.initheight + 0.1

    def env_metadata(self):
        metadata = self.arena_metadata
        metadata.update({
            "object": self.object.name,
        })
        return metadata