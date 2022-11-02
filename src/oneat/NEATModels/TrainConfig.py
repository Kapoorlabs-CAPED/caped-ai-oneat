#!/usr/bin/env python3

import argparse


class TrainConfig(argparse.Namespace):
    def __init__(
        self,
        cell_type_name,
        cell_type_label,
        cell_position_name,
        cell_position_label,
        **kwargs
    ):

        self.cell_type_name = cell_type_name
        self.cell_type_label = cell_type_label
        self.cell_position_name = cell_position_name
        self.cell_position_label = cell_position_label
        assert len(cell_type_name) == len(cell_type_label)
        assert len(cell_position_name) == len(cell_position_label)

    def to_json(self):

        config = {}
        configCord = {}
        for i in range(0, len(self.cell_type_name)):

            config[self.cell_type_name[i]] = self.cell_type_label[i]
        for i in range(0, len(self.cell_position_name)):

            configCord[self.cell_position_name[i]] = self.cell_position_label[
                i
            ]

        return config, configCord
