**Teaching Robots to Span the Space of Functional Expressive Motion** Code Release

File structure

**cassie_blender**: Used for visualizing full body Cassie trajectories along with obstacles and goals in Blender
- _Cassie.blend_: Main blender file, 95MB so not in github, get from [gdrive](https://drive.google.com/file/d/1bMBQcnQ9mz0jz2RVG-koAfvYmA8TPT_Q/view?usp=sharing)
- _external_motion_loader.py_ used to load waypoints from a motion.csv and obstacles and goals from a task_dict.pkl, both in a folder you specify in this file (e.g. test_viz_folder)

**emotive_cassie_opti**: Used for optimizing the trajectories cassie_blender visualizes.
- _dynamics/SLIP_opti.py_ is the main file which optimizes trajectories. It relies on other files in this trajectories for cassie specifications, gets trajectory optimization orders (start/goal, obstacles, & target style) from another process, and leverages _kinematics/style_cost_gen.py_ to include style_cost in its optimization.

**robot_style_cost**: Main folder containing the core algorithm and the infrastructure for connecting it with the Cassie specific code as well as for VacuumBot.
- _cart_: Contains VacuumBot specific code. To launch VacuumBot experiments run _cart/cart_interface.py_, see _interface.py_ for more information.
- _full_cassie_: Contains Cassie specific code. To launch Cassie experiments run _full_cassie/full_cassie_interface.py_, see _interface.py_ for more information.
- _interface.py_: Main launching point for all experiments. May be used to reproduce simulated/real human experiments from the paper, as well as analyze results. Leverages _style_net.py_ to manage style cost and optimizes trajectories. Trajectory optimization is done by _traj_opt.py_ for VacuumBot and exported to _SLIP_opti.py_ for Cassie which is expected to be running in another process.
- _style_net.py_: Manages the style discriminator from _traj_cost_model.py_ and the language model from _language_model.py_
- _env.py_: Base environment abstraction that if respected allows our method to easily generalize to other robots.

**shared**: Used for some general files that are commonly referenced across the project. In addition to below will be used for communication between robot_style_cost and emotive_cassie_opti & cassie_blender during the trajectory optimization and visualization process.
- **vad_ref.pkl**: word VAD lookup table.
- **actionables.pkl**: Set of easy to enact style words
- **sm_params.pkl**: Sentence model parameters, 418MB so not in github, get from [gdrive](https://drive.google.com/file/d/1pnG8z3sD8Umrco2apLK2lUt_N5UAuFCs/view?usp=sharing)
