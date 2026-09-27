#include <WiFi.h>
#include "esp_wifi.h"

void csi_callback(void *ctx, wifi_csi_info_t *data)
{
  if (!data) return;

  Serial.print("CSI:");

  for (int i = 0; i < data->len; i++) {
    Serial.print((int8_t)data->buf[i]);
    if (i < data->len - 1) Serial.print(",");
  }
  Serial.println();
}

void setup()
{
  Serial.begin(115200);
  delay(1000);

  WiFi.mode(WIFI_MODE_STA);
  WiFi.disconnect(true, true);

  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);

  wifi_csi_config_t conf = {
      .lltf_en = 1,
      .htltf_en = 1,
      .stbc_htltf2_en = 1,
      .ltf_merge_en = 1,
      .channel_filter_en = 1,
      .manu_scale = 0,
      .shift = 0
  };
  esp_wifi_set_csi_config(&conf);
  esp_wifi_set_csi_rx_cb(csi_callback, NULL);
  esp_wifi_set_csi(1);

  Serial.println("CSI Ready…");
}

void loop()
{
}
