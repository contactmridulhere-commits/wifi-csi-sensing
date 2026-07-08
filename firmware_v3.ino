/*
 * CSI Sensing Firmware v3  ·  ESP32
 * ---------------------------------
 * Upgrades:
 *   - Active null-frame injection (self-generates CSI, no reliance on ambient traffic)
 *   - I/Q streaming with frame counter + microsecond timestamp
 *   - Role select: TX beacon node OR RX sensing node (for multi-node arrays)
 *   - Configurable channel / injection rate
 *
 * Serial format (v2, parsed by csi_engine):
 *   CSI:<frame>,<timestamp_us>,<num_bytes>:I0,Q0,I1,Q1,...
 *
 * MULTI-NODE SETUP (the real path to localization):
 *   - Flash ONE board as ROLE_TX  (continuous beacon)
 *   - Flash N boards as ROLE_RX   (sensing, one per serial port on the host)
 *   - Place RX nodes around the monitored area at different angles.
 *   - Add each RX port to CONFIG.serial_ports in config.py.
 */

#include <WiFi.h>
#include "esp_wifi.h"
#include "esp_timer.h"

// ─── CONFIG ──────────────────────────────────────────
#define ROLE_RX            1     // set to 0 for a dedicated TX beacon node
#define CSI_CHANNEL        1
#define SERIAL_BAUD        115200
#define INJECT_INTERVAL_MS 20    // 50 Hz
// ─────────────────────────────────────────────────────

static uint32_t frame_count = 0;

static void promisc_rx_cb(void *buf, wifi_promiscuous_pkt_type_t type) {
  // Reception must be enabled for CSI to fire; no per-packet work needed here.
}

void csi_callback(void *ctx, wifi_csi_info_t *data) {
  if (!data || !data->buf || data->len == 0) return;
  frame_count++;
  int64_t ts_us = esp_timer_get_time();
  Serial.printf("CSI:%lu,%lld,%d:", frame_count, ts_us, data->len);
  for (int i = 0; i < data->len; i++) {
    Serial.print((int8_t)data->buf[i]);
    if (i < data->len - 1) Serial.print(",");
  }
  Serial.println();
}

static void inject_null_frame() {
  static uint8_t f[] = {
    0x48, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00
  };
  esp_wifi_80211_tx(WIFI_IF_STA, f, sizeof(f), false);
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(1000);

  WiFi.mode(WIFI_MODE_STA);
  WiFi.disconnect(true, true);
  delay(100);

  esp_wifi_set_promiscuous(true);
  esp_wifi_set_promiscuous_rx_cb(promisc_rx_cb);
  esp_wifi_set_channel(CSI_CHANNEL, WIFI_SECOND_CHAN_NONE);

#if ROLE_RX
  wifi_csi_config_t conf = {
    .lltf_en = 1, .htltf_en = 1, .stbc_htltf2_en = 1,
    .ltf_merge_en = 1, .channel_filter_en = 0,
    .manu_scale = 0, .shift = 0
  };
  esp_wifi_set_csi_config(&conf);
  esp_wifi_set_csi_rx_cb(csi_callback, NULL);
  esp_wifi_set_csi(true);
  Serial.println("CSI_V3_READY_RX");
#else
  Serial.println("CSI_V3_READY_TX");
#endif
  Serial.printf("# ch=%d inject=%dms role=%s\n",
                CSI_CHANNEL, INJECT_INTERVAL_MS, ROLE_RX ? "RX" : "TX");
}

void loop() {
  inject_null_frame();          // both roles inject to keep the link stimulated
  delay(INJECT_INTERVAL_MS);
}
