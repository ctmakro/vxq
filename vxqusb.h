/*
 * Copyright (c) 2006-2020, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2020-12-20     luosh       the first version
 */
#ifndef VXQUSB_H_
#define VXQUSB_H_

#include <stdbool.h>
#include <stdint.h>
#include "vxqconfig.h"

#define USB_MSG_SYS_REPORT     0
//#define USB_MSG_DEBUG_WRITE  1
#define USB_MSG_CMD_ACK        2
#define USB_MSG_JOINT          3
#define USB_MSG_MOTION         4
#define USB_MSG_PARAM_LIST     5
#define USB_MSG_PARAM_ACK      6
#define USB_MSG_SERIAL_INFO    7
#define USB_MSG_SERIAL_READ    8
#define USB_MSG_IR             9
//#define USB_MSG_RF_READ      10
#define USB_MSG_JREGACK        11
#define USB_MSG_LIMITS         12

#define USB_CMD_SYS_REQ        0
//#define USB_CMD_DEBUG_READ   1
#define USB_CMD_MOTION         2
#define USB_CMD_PARAM_REQ      3
#define USB_CMD_PARAM_SET      4
#define USB_CMD_SERIAL_REQ     5
#define USB_CMD_SERIAL_WRITE   6
//#define USB_CMD_RF_WRITE     7
#define USB_CMD_JREGACCESS     8
#define USB_CMD_LIMITS         9

#define USB_MSG_MOTION_S_RUNNING      1
#define USB_MSG_MOTION_S_REACHED      2
#define USB_MSG_MOTION_S_CARTESIAN    4

#define USB_CMD_SYS_REQ_REPORT        0
#define USB_CMD_SYS_REQ_REBOOT        1
#define USB_CMD_SYS_REQ_UPGRADE       2

#define USB_CMD_ERR_NOCMD             1         //no such command
#define USB_CMD_ERR_LEN               2         //length mismatch

#define USB_CMD_MOTION_MODE_READBACK  0
#define USB_CMD_MOTION_MODE_GO        1
#define USB_CMD_MOTION_MODE_STEP      2
#define USB_CMD_MOTION_MODE_STOP      3

#define USB_CMD_MOTION_F_CARTESIAN    0x01
#define USB_CMD_MOTION_F_SPEEDSET     0x02

#define USB_CMD_LIMITS_F_SPEEDSET     0x01
#define USB_CMD_LIMITS_F_VELMAX       0x02
#define USB_CMD_LIMITS_F_ACCMAX       0x04
#define USB_CMD_LIMITS_F_JERKMAX      0x08
#define USB_CMD_LIMITS_F_PERSIST      0x40
#define USB_CMD_LIMITS_F_GET          0x80

#define USB_CMD_PARAM_REQ_R_VAL         0       //param value
#define USB_CMD_PARAM_REQ_R_DEFVAL      1       //param default value
#define USB_CMD_PARAM_REQ_R_IDLIST      2       //param id list

struct usb_header_s {
    uint32_t ts;
    uint8_t type;
};

struct usb_cmd_sys_request_s {
    uint32_t ts;
    uint8_t type;
    uint8_t req;
};

struct usb_cmd_debug_s {
    uint32_t ts;
    uint8_t type;
};

struct usb_cmd_motion_s {
    uint32_t ts;
    uint8_t type;
    uint8_t mode;
    //0 -> readback, 1 -> go, 2 -> step, 3 -> stop
    uint8_t flags;
    //bit0 = joint(0) / cartesian(1), bit1 = set speedset
    uint8_t speedset;
    //max velocity percentage, 10 ~ 100
    float target[NJOINTS];
    //target position
};  //40B

struct usb_cmd_param_req_s {
    uint32_t ts;
    uint8_t type;
    uint8_t req;
    //0 -> param val, 1 -> default val, 2 -> id list
    uint16_t id;
};

struct usb_cmd_param_set_s {
    uint32_t ts;
    uint8_t type;
    uint8_t len;
    uint16_t id;
    uint8_t val[56];
};

struct usb_cmd_serial_req_s {
    uint32_t ts;
    uint8_t type;
    uint8_t len;
    //0 -> info req, >0 -> read req
};

struct usb_cmd_serial_write_s {
    uint32_t ts;
    uint8_t type;
    uint8_t len;
    uint8_t val[58];
};

struct usb_cmd_limits_s {
    uint32_t ts;
    uint8_t type;
    uint8_t flags;
    //bit0 = speedset, bit1 = velmax, bit2 = accmax, bit3 = jerkmax, bit7 = get(1)/set(0)
    float x[NJOINTS];
};//38B

struct usb_msg_sys_report_s {
    uint32_t ts;
    uint8_t type;
    uint8_t len;
    char version[20];
    uint8_t errcode;
};

struct usb_msg_cmd_ack_s {
    uint32_t ts;
    uint8_t type;
    uint8_t cmd;
    int16_t err;    //rt_err_t
    //0 = success, ENOSYS = no such command
};

struct usb_msg_debug_write_s {
    uint32_t ts;
    uint8_t type;
    uint8_t rsvd[0];
};

struct usb_msg_param_list_s {
    uint32_t ts;
    uint8_t type;
    uint8_t subframe;

    union usb_msg_param_list_u_u {
        struct usb_msg_param_list_s0_s {
           char version[20];
           uint32_t offset;
           uint32_t size;
           uint16_t nitems;
        } s0;   //30B

        struct usb_msg_param_list_sn_s {
            uint16_t id;    //0xffff -> invalid item
            uint16_t addr;
            uint8_t type;
            uint8_t flag;
            uint8_t sz;
            uint8_t n;
        } s[7];   //8B
    } u;
};   //62B

#define USB_MSG_PARAM_ACK_F_DEFAULT     1

struct usb_msg_param_ack_s {
    uint32_t ts;
    uint8_t type;
    uint8_t len;
    uint16_t id;
    uint8_t flags;  //bit0 = default
    uint8_t val[55];
};  //64B

struct usb_msg_joint_data_s {
    uint8_t options;
    uint8_t status;
    uint16_t pos;
    uint16_t fb;
    uint16_t current;
    uint8_t mode;
    int8_t temp;
    //uint8_t ack;
    //uint8_t rsvd;
};  //10B

struct usb_msg_joint_s {
    uint32_t ts;
    uint8_t type;
    uint8_t subframe;   //0 = j1~j5, 1 = j6-j8

    union usb_msg_joint_u_u {
        struct usb_msg_joint_s0_s {
            struct usb_msg_joint_data_s joint[5];
        } s0;   //50B

        struct usb_msg_joint_s1_s {
            struct usb_msg_joint_data_s joint[3];
            uint8_t motion_status; //bit0 = running, bit1 = reached
            uint8_t speedset;
            float lastpos[NJOINTS];
        } s1;   //56B
    } u;
};  //62B

struct usb_msg_motion_s {
    uint32_t ts;
    uint8_t type;
    uint8_t subframe;

    uint8_t motion_status;
    //bit0 = running, bit1 = reached, bit2 = cartesian_valid
    uint8_t speedset;
    float last_joint_pos[NJOINTS];
    float last_cartesian_pos[NJOINTS];
};  //64B

struct usb_msg_serial_info_s {
    uint32_t ts;
    uint8_t type;
    uint8_t rsvd;
    uint16_t tx_space;
    uint16_t rx_avail;
};

struct usb_msg_serial_read_s {
    uint32_t ts;
    uint8_t type;
    uint8_t len;
    uint8_t val[58];
};

struct usb_msg_ir_s {
    uint32_t ts;
    uint8_t type;
    uint8_t n;
    struct usb_msg_ir_event_s {
        uint16_t repeat;    //0xffff -> key up, 0~0xfffe -> key down
        uint16_t keycode;
    } event[14];
};

struct usb_msg_rf_s {
    uint32_t ts;
    uint8_t type;
};

struct usb_msg_limits_s {
    uint32_t ts;
    uint8_t type;
    uint8_t flags;
    //bit0 = speedset, bit1 = velmax, bit2 = accmax, bit3 = jerkmax
    float x[NJOINTS];
};

#endif /* VXQUSB_H_ */
