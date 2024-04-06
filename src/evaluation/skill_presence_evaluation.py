import sys
import os
import re
import json
import pandas as pd
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger('skill_presence_evaluation')
logger.setLevel(logging.INFO)

sys.path.append(os.path.join(sys.path[0], '..'))
sys.path.append(os.path.join(sys.path[0], '../..'))

from .evaluation import Evaluation

dir_path = os.path.dirname(os.path.realpath(__file__))

DBT_SKILLS_REF = pd.read_csv(
    os.path.join(dir_path, '..', 'data', 'dbt_skills_ref.csv'),
    index_col='skill_id',
)
VALID_SKILL_IDS = list(DBT_SKILLS_REF.index)
with open(os.path.join(dir_path, '..', 'data', 'skill_presence_regex.json'), 'r') as f:
    SKILL_PRESENCE_REGEX = json.load(f)


class DBTSkill:
    def __init__(self, skill_id: str):
        assert skill_id in VALID_SKILL_IDS
        self.skill_id = skill_id

        ref_row = DBT_SKILLS_REF.loc[skill_id]
        self.skill_name = ref_row.skill_name
        self.module_focus = ref_row.module_focus
        self.module_name = ref_row.module_name
        self.category_name = None
        category_name = ref_row.category_name
        if pd.notna(category_name):
            self.category_name = category_name

        self.skill_desc = ref_row.skill_desc
        self.pattern_str = SKILL_PRESENCE_REGEX[self.skill_id]
        self.pattern = re.compile(self.pattern_str)

    def get_hierarchy_match(self, other):
        if other is None:
            return 'no_skill_detected', 0
        if self.skill_name == other.skill_name:
            return 'skill', 4
        if self.category_name == other.category_name:
            return 'category', 3
        if self.module_name == other.module_name:
            return 'module', 2
        return 'no_match', 1

    def get_regex_result(self, conversation: str):
        if re.search(self.pattern, conversation):
            return 1
        return 0


class DBTSkills:
    def __init__(self, skill_ids: list[str] = None):
        if not skill_ids:
            self.skill_ids = VALID_SKILL_IDS
        else:
            self.skill_ids = skill_ids
        self.skills = [DBTSkill(skill_id=skill_id) for skill_id in self.skill_ids]
        self.skills_dict = {skill.skill_id: skill for skill in self.skills}

    def get_skill(self, skill_id: str):
        return self.skills_dict.get(skill_id)

    def get_regex_results(self, conversation: str):
        return self.skills, [skill.get_regex_result(conversation) for skill in self.skills]


class SkillPresenceEvaluation(Evaluation):

    name = 'Skill Presence'

    def __init__(self, skill_ids: list[str] = None):
        self.dbt_skills = DBTSkills(skill_ids)
        logger.info(f'Launching skill presence evaluation with following skill ids: {self.dbt_skills.skill_ids}')

    @staticmethod
    def _check_ids_all_present(skill_ids, search_skill_ids):
        return all([search_id in skill_ids for search_id in search_skill_ids])

    def run_evaluation(self, conversation: str):
        skill_ids_present = []
        skills, regex_results = self.dbt_skills.get_regex_results(conversation)
        for i, res in enumerate(regex_results):
            if res == 1:
                skill_ids_present.append(skills[i].skill_id)

        if self._check_ids_all_present(skill_ids_present, ['M1', 'DT4']):
            # Remove "Wise Mind" if "Distracting with Wise Mind ACCEPTS" is present
            skill_ids_present.remove('M1')
        if self._check_ids_all_present(skill_ids_present, ['M2', 'DT1']):
            # Remove "Observe" if "STOP" is present
            skill_ids_present.remove('M2')
        if self._check_ids_all_present(skill_ids_present, ['M3', 'IE2']):
            # Remove "Describe" if "DEAR MAN" is present
            skill_ids_present.remove('M3')
        if self._check_ids_all_present(skill_ids_present, ['IE10', 'IE3']):
            # Remove "Validation" if "GIVE" is present
            skill_ids_present.remove('IE10')

        return skill_ids_present
