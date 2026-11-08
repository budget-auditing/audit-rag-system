#!/usr/bin/env python3
"""
证据5元组解析器：将证据文本解析为 (t, d, m, span, meta) 5元组
"""

import re
from typing import Dict, List, Optional


class EvidenceParser:
    """证据解析器：将文本解析为5元组"""
    
    def __init__(self):
        # 年份模式
        self.year_patterns = [
            r'20\d{2}年',  # 2023年
            r'20\d{2}',     # 2023
        ]
        
        # 部门关键词（常见部门名称）
        self.department_keywords = [
            '财政厅', '财政局', '财政部', '发改委', '教育厅', '卫生厅',
            '交通厅', '农业厅', '工信厅', '公安厅', '民政厅', '人社厅',
            '科技厅', '科技局',
            '省级', '市级', '县级', '本级', '政府', '机关', '事业单位',
            '人大', '政协', '法院', '检察院'
        ]
        
        # 指标关键词（财政指标）
        self.metric_keywords = {
            'budget': ['预算', '编制', '安排', '计划'],
            'revenue': ['收入', '税收', '非税'],
            'expenditure': ['支出', '开支', '花费'],
            'transfer': ['转移支付', '补助', '拨付', '下达'],
            'debt': ['债务', '专项债', '一般债', '借款'],
            'execution': ['执行', '完成', '进度'],
            'performance': ['绩效', '效果', '成效', '评估'],
            'balance': ['结余', '结转', '平衡'],
        }
        
        # 口径注记关键词
        self.caliber_keywords = [
            '含下属单位', '不含下属单位', '本级',
            '合并', '并表', '单体',
            '调整后口径', '可比口径', '同口径',
            '纳入', '并入', '剔除', '不含',
            '政府性基金', '一般公共预算', '国有资本经营预算',
            '调整', '变更', '重新统计'
        ]
    
    def parse_evidence(self, text: str) -> Dict:
        """
        将证据文本解析为5元组
        
        Args:
            text: 证据文本
            
        Returns:
            {
                't': 年份,
                'd': 部门,
                'm': 指标列表,
                'span': 原文,
                'meta': 口径注记列表
            }
        """
        return {
            't': self.extract_year(text),
            'd': self.extract_department(text),
            'm': self.extract_metrics(text),
            'span': text,
            'meta': self.extract_caliber_meta(text)
        }
    
    def extract_year(self, text: str) -> Optional[str]:
        """提取年份"""
        for pattern in self.year_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # 返回第一个匹配的年份，去掉"年"字
                year = matches[0].replace('年', '')
                return year
        return None
    
    def extract_department(self, text: str) -> Optional[str]:
        """提取部门"""
        for dept in self.department_keywords:
            if dept in text:
                return dept
        return None
    
    def extract_metrics(self, text: str) -> List[str]:
        """提取指标"""
        metrics = []
        for metric_type, keywords in self.metric_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    metrics.append(metric_type)
                    break  # 每个类型只添加一次
        return metrics
    
    def extract_caliber_meta(self, text: str) -> List[str]:
        """提取口径注记"""
        meta = []
        for caliber_kw in self.caliber_keywords:
            if caliber_kw in text:
                meta.append(caliber_kw)
        return meta
    
    def parse_evidence_set(self, evidence_texts: List[str]) -> List[Dict]:
        """批量解析证据集"""
        return [self.parse_evidence(text) for text in evidence_texts]
    
    def format_evidence_tuple(self, evidence: Dict) -> str:
        """格式化显示5元组"""
        t = evidence['t'] or '未知年份'
        d = evidence['d'] or '未指定部门'
        m = ', '.join(evidence['m']) if evidence['m'] else '未识别指标'
        span = evidence['span'][:50] + '...' if len(evidence['span']) > 50 else evidence['span']
        meta = ', '.join(evidence['meta']) if evidence['meta'] else '无口径注记'
        
        return f"""
证据5元组:
  t (年份): {t}
  d (部门): {d}
  m (指标): {m}
  span (文本): {span}
  meta (口径): {meta}
"""


def check_caliber_consistency(evidence_set: List[Dict]) -> Dict:
    """
    检查证据集的口径一致性
    
    Returns:
        {
            'consistent': bool,
            'issues': List[str],
            'year_range': Set[str],
            'departments': Set[str],
            'caliber_conflicts': List[str]
        }
    """
    years = set()
    departments = set()
    caliber_metas = []
    issues = []
    
    for evidence in evidence_set:
        if evidence['t']:
            years.add(evidence['t'])
        if evidence['d']:
            departments.add(evidence['d'])
        if evidence['meta']:
            caliber_metas.extend(evidence['meta'])
    
    # 检查年份一致性
    if len(years) > 1:
        issues.append(f"包含多个年份: {', '.join(sorted(years))}")
    
    # 检查口径冲突
    caliber_conflicts = []
    if '含下属单位' in caliber_metas and '不含下属单位' in caliber_metas:
        caliber_conflicts.append('单位范围冲突: 同时含/不含下属单位')
    if '调整后口径' in caliber_metas and '调整前口径' in caliber_metas:
        caliber_conflicts.append('口径版本冲突: 调整前后口径混用')
    if '合并' in caliber_metas and '单体' in caliber_metas:
        caliber_conflicts.append('报表口径冲突: 合并/单体混用')
    
    if caliber_conflicts:
        issues.extend(caliber_conflicts)
    
    return {
        'consistent': len(issues) == 0,
        'issues': issues,
        'year_range': years,
        'departments': departments,
        'caliber_conflicts': caliber_conflicts
    }

