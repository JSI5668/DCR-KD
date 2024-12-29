import torch
import torch.nn as nn


class DynamicAttentionWeights(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(DynamicAttentionWeights, self).__init__()
        # Teacher와 Student Attention Map에 대한 채널별 학습 가능한 가중치 초기화
        self.alpha_teacher = nn.Parameter(torch.ones(1, num_channels_in, 1, 1))  # (1, C_in, 1, 1)
        self.alpha_student = nn.Parameter(torch.ones(1, num_channels_in, 1, 1))  # (1, C_in, 1, 1)

        # 1x1 Convolution 레이어
        self.conv_teacher = nn.Conv2d(num_channels_in, num_channels_out, kernel_size=1, bias=False)
        self.conv_student = nn.Conv2d(num_channels_in, num_channels_out, kernel_size=1, bias=False)

    def forward(self, attention_map_teacher, attention_map_student):
        """
        채널별 학습 가능한 가중치를 적용하고, 채널 수를 맞춤
        Args:
            attention_map_teacher: Teacher의 Channel Attention Map (B, C_in, 1, 1)
            attention_map_student: Student의 Channel Attention Map (B, C_in, 1, 1)
        Returns:
            weighted_teacher_attention: Teacher Attention Map (B, C_out, 1, 1)
            weighted_student_attention: Student Attention Map (B, C_out, 1, 1)
        """
        # 동일한 디바이스로 이동
        device_teacher = attention_map_teacher.device
        device_student = attention_map_student.device

        self.conv_teacher = self.conv_teacher.to(device_teacher)
        self.conv_student = self.conv_student.to(device_student)

        alpha_teacher = self.alpha_teacher.to(device_teacher)
        alpha_student = self.alpha_student.to(device_student)

        # 채널별 학습 가능한 가중치 적용
        weighted_teacher_attention = alpha_teacher * attention_map_teacher  # (B, C_in, 1, 1)
        weighted_student_attention = alpha_student * attention_map_student  # (B, C_in, 1, 1)

        # 1x1 Convolution으로 채널 수 변환
        weighted_teacher_attention = self.conv_teacher(weighted_teacher_attention)  # (B, C_out, 1, 1)
        weighted_student_attention = self.conv_student(weighted_student_attention)  # (B, C_out, 1, 1)

        return weighted_teacher_attention, weighted_student_attention




