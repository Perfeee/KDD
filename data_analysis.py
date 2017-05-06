import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
import glob
from matplotlib import pyplot as plt


train_data_file_prefix = "../dataSets/training/"
gen_data_file_prefix = "../dataSets/gen_data/"
def link_time_ave_analysis(traj_df, using_file=True, name=""):
    # 按link_id 进行统计， traj_df是广义的，只要是一个df就行。一般是analysis_by_time的子函数。
    analysis_log = ""
    link_time_dict_path = gen_data_file_prefix + name + "_"+"link_time_dict.pkl"
    if glob.glob(link_time_dict_path) and using_file:
        with open(link_time_dict_path, "rb") as f:
            link_dict = pickle.load(f)
    else:
        link_dict = defaultdict(list)
        for row in traj_df["travel_seq"]:
            for link in row.split(";"):
                tmp = link.split("#")
                link_dict[tmp[0]].append(tmp[2])
    if using_file:
        with open(link_time_dict_path, "wb") as f:
                pickle.dump(link_dict, f)
    for link_id in links["link_id"]:
        tmp = np.array(link_dict[str(link_id)]).astype(float)
        #一致性（一刀切）剔除上界和下界分别1%的异常数据
        tmp = tmp[np.where(tmp < np.percentile(tmp, 99))]
        tmp = tmp[np.where(tmp > np.percentile(tmp, 1))]

        single_result = "link_id:{}, mean:{}, max:{}, min:{}, std:{}, coefficient of variation: {}".format(link_id, tmp.mean(), tmp.max(), tmp.min(), tmp.std(), tmp.std()/tmp.mean())
        print(single_result)
        analysis_log += single_result + "\n"
    return analysis_log

def train_local_data_gen(train, link_id, target_routes, routes):
    # 用于抽取数据，两个过滤标准，一是路线，而是linkid， 暂时现在是路线可以多条，linkid只能有一个。如果要按全部路线进行分析，可以使用analysis_by_time函数
    # extract linkid 123, route a2, a3
    intersection_id, tollgate_id = target_routes[:,0], target_routes[: , 1]
    extra_condition = np.array([0])
    for intersection, tollgate in zip(intersection_id, tollgate_id):
        if extra_condition.any():
            extra_condition = ((train.intersection_id == intersection) & (train.tollgate_id == tollgate)) | extra_condition
        else:
            extra_condition = ((train.intersection_id == intersection) & (train.tollgate_id == tollgate))
    train = train[extra_condition]
    local_data = []
    def compute_if_link_id_miss(i):
        for j in i.split(";"):
            pass
    for i in train.travel_seq:
        if link_id not in i:
            pass
        for j in i.split(";"):
            if j[0:3] == link_id:
                tmp = j.split("#")
                local_data.append(tmp)
                break
    return local_data

def analysis_by_time(traj_df, group_basis, using_file = False):
    # 可根据分组对数据分类分析,分组的group_basis必须是traj_df中的列名
    group_date = traj_df.groupby(by=group_basis)
    result_log = ""
    for name, group in group_date:
        print(name)
        result_log += str(name) + "\n"
        single_log = link_time_ave_analysis(group,using_file=using_file, name=group_basis + "_" +str(name))
        result_log += single_log
    with open(gen_data_file_prefix + group_basis +"_analysis_log.txt", "w") as f:
        f.write(result_log)

def fill_if_link_id_missing(train, routes, links):
    # 填补路线中，观测点缺失数据
    # 缺失数据不会是该路线中的最后一个linkid和第一个linkid
    # 有两个记录，分别是57942， 105291缺了两段（不止两个link）的数据
    # 这个填充方式有两个缺陷，一是占比问题，二是精度问题，只能精确到秒
    def ratio_compute(miss_sub_route, links):
        # 占比计算
        # miss_sub_root 是一个linkid串，注意跟下面的subroute不一样，下面的linkid的位置串
        tmp = []
        for linkid in miss_sub_route:
            tmp.append(links[links.link_id == linkid]["length"].values[0]/links[links.link_id == linkid]["lanes"].values[0])
        s = sum(tmp)
        tmp = [x/s for x in tmp]
        return tmp

    for i in range(train.shape[0]):
        j = train.iloc[i,:]
        route = routes[(routes.intersection_id == j.intersection_id) & (routes.tollgate_id == j.tollgate_id)]
        route_link_seq = route["link_seq"].values[0].split(",")
        tmp = j.travel_seq.split(";")
        diff = set(route_link_seq).difference(set([x[0:3] for x in tmp]))
        # diff = list(diff)

        if len(diff) > 0:
            pos = [route_link_seq.index(x) for x in diff]
            pos = sorted(pos)
            # print(diff)
            # print(pos)
            if (pos[-1] - pos[0]) != (len(pos)-1):
                # 判断是否多个连续缺失
                for i in range(len(pos)-1, 0, -1):
                    #寻找大于两段缺失段的缺失点
                    if (pos[0:i][-1] -pos[0:i][0]) != (len(pos[0:i])-1):
                        continue
                    else:
                        #找到切割点
                        sub_route1 = pos[0:i]
                        sub_route2 = pos[i:]
                        missing_sub_route = [sub_route1, sub_route2]
                        break
            else:
                missing_sub_route = [pos]
            for sub_route in missing_sub_route:
                # print(sub_route)
                # print(route_link_seq[sub_route[0]-1])
                # print("right", route_link_seq[sub_route[-1]+1])
                # print(j)
                # print(j.travel_seq)
                left = [x for x in tmp if route_link_seq[sub_route[0]-1] in x]
                right = [x for x in tmp if route_link_seq[sub_route[-1]+1] in x]
                # print(left)
                # print(right)
                miss_sum_time_length = (pd.to_datetime(right[0].split("#")[1]) - pd.to_datetime(left[0].split("#")[1])).seconds
                sub_route_link_id = []
                for x in sub_route:
                    sub_route_link_id.append(route_link_seq[x])
                ratio_list = ratio_compute(sub_route_link_id, links)
                insert_seq = []
                cum_ratio = 0
                for link, ratio in zip(sub_route_link_id, ratio_list):
                    cum_ratio += ratio
                    insert_string = ""
                    insert_string += link
                    insert_string += "#"
                    insert_string += str(pd.to_datetime(tmp[sub_route[0] -1].split("#")[1]) + pd.to_timedelta(str(miss_sum_time_length*cum_ratio)+"S"))
                    insert_string += "#" + str(miss_sum_time_length*ratio)
                    insert_seq.append(insert_string)
                tmp = tmp[0:sub_route[0]-1] + insert_seq + tmp[sub_route[0]-1:]
            new_seq = ";".join(tmp)
            train.iloc[i,4] = new_seq
            # print("new_seq", new_seq)
        else:
            continue
            # p = [s+1 for s in pos]
            # if p[0:-1] != pos[1:]:
            #     print(route, j.intersection_id, j.tollgate_id)
            #     print(i, route_link_seq, tmp, diff)
            #     print(pos[1:], p[0:-1])
    return train


if __name__ == "__main__":
    routes = pd.read_csv(train_data_file_prefix + "routes(table4).csv")
    links = pd.read_csv(train_data_file_prefix + "links(table3).csv")
    links["link_id"] = links.link_id.astype(str)
    if glob.glob(train_data_file_prefix + "trajectories(new)_training.csv"):
        pd.read_csv(train_data_file_prefix + "trajectories(new)_training.csv")
    else:
        traj_df = pd.read_csv(train_data_file_prefix + "trajectories(table5)_training.csv")
        traj_df = fill_if_link_id_missing(traj_df, routes, links)
        traj_df.to_csv(train_data_file_prefix + "trajectories(new)_training.csv")

    # print(traj_df.columns)
    # traj_df["starting_time"] = pd.to_datetime(traj_df["starting_time"])
    # traj_df["starting_date"] = traj_df["starting_time"].map(pd.datetime.date)
    # traj_df["starting_hour"] = traj_df["starting_time"].dt.hour
    # traj_df["starting_weekday"] = traj_df["starting_time"].dt.weekday
    # group_basis = "starting_date"
    # group_basis = "starting_hour"
    # group_basis = "starting_weekday"
    # analysis_by_time(traj_df=traj_df, group_basis=group_basis)
    # link123_local_data = train_local_data_gen(traj_df, "123", ["A","A"], tollgate_id=[2, 3])
    # link123_local_data = np.array(link123_local_data)
    # link123_local_data.dump(gen_data_file_prefix + "link_123.pkl")
