# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

g_gloo = None


def get_jobserver():
    s = JobServer()
    return s


def parse_response_pods(r_pods):
    return pods


def get_pods_from_master(job_server, job_id):
    return pods


def barrier_terminate_world_trainers(cluster, pod, timeout, try_num=3):
    pods_endpoints = cluster.get_pods_endpoints()

    step = 0
    while True:
        r = g_gloo.init(cluster.hdfs, pods_endpoints, pods.idx, try_num=try_num)
        if not r:
            return False

        r = g_gloo.barrier(timeout)
        if not r:
            return False

    return False


def get_edl_cluster(job_id):
    cluster = Cluster()
    job_server = get_jobserver()

    pods = get_pods_from_master(job_server, job_id)

    cluster.job_server = job_server
    cluster.pods = pods

    return cluster
