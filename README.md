# classification_model_mlops

# Grafana Installation & Usage on Windows (Local)

This guide explains how to install Grafana on Windows, start it, stop it, and run it locally.

---

## 1. Install Grafana on Windows

1. Download Grafana from the official website:
   - Choose **Windows Installer (64-bit)**
2. Run the `.msi` installer.
3. During setup:
   - ✅ Keep **Grafana Enterprise** selected  
     > This installs Grafana locally. No license or cloud usage is required.
   - ⬜ Optional: Uncheck **Run Grafana as a Service** if you want to control start/stop manually.
4. Complete the installation.

---

## 2. Start Grafana

### Option A: Start Grafana as a Windows Service

If you installed Grafana as a service:

1. Press **Win + R**
2. Type:
```
   services.msc
```
3. Find **Grafana** or **Grafana Server**
4. Right-click → **Start**

Open in browser:
```
http://localhost:3000
```

---

### Option B: Start Grafana Manually (Recommended for Local Development)

1. Open **PowerShell**
2. Run:
```powershell
   cd "C:\Program Files\GrafanaLabs\grafana\bin"
   .\grafana-server.exe
```
   Keep the terminal open.

Open in browser:
```
http://localhost:3000
```

---

## 3. Login to Grafana

Default credentials:
```
Username: admin
Password: admin
```
You will be prompted to change the password.

---

## 4. Stop Grafana

### If Running Manually

Press:
```
CTRL + C
```

### If Running as a Windows Service

**Using Services UI:**
1. Press **Win + R**
2. Type:
```
   services.msc
```
3. Find **Grafana**
4. Right-click → **Stop**

**Using Command Line (Admin):**
```powershell
net stop grafana
```

---

## 5. Start Grafana Again (After Stopping)

### Using Services UI

1. Press **Win + R**
2. Type:
```
   services.msc
```
3. Find **Grafana**
4. Right-click → **Start**

### Using Command Line (Admin)
```powershell
net start grafana
```

---

## 6. Verify Grafana is Running

Open browser:
```
http://localhost:3000
```
If the page loads, Grafana is running successfully.

---

## Notes

- Grafana runs completely locally.
- No Grafana Cloud account is required.
- Enterprise features remain inactive unless a license is added.
- Default port used is **3000**.


# Install & Run Prometheus on Windows (Local)

This guide explains how to install Prometheus on Windows, configure it using a YAML file, and run it locally.

---

## 1. Download Prometheus for Windows

1. Go to the official Prometheus download page.
2. Download **Windows (64-bit)** ZIP file:
```
   prometheus-<version>.windows-amd64.zip
```

---

## 2. Extract Prometheus

1. Right-click the downloaded ZIP file → **Extract All**
2. Move the extracted folder to:
```
   C:\prometheus\
```

Folder structure should look like:
```
C:\prometheus
├── prometheus.exe
├── promtool.exe
├── prometheus.yml
└── consoles\
```

---

## 3. Configure Prometheus (Edit YAML)

Open the file:
```
C:\prometheus\prometheus.yml
```

### Example Basic Configuration
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]
```

Save the file.

---

## 4. Run Prometheus

### Open PowerShell

Navigate to Prometheus folder:
```powershell
cd C:\prometheus
```

Start Prometheus:
```powershell
.\prometheus.exe --config.file=prometheus.yml
```

---

## 5. Open Prometheus Web UI

Open browser:
```
http://localhost:9090
```

---

## 6. Verify Configuration (Optional)

Before running Prometheus, you can validate the YAML file:
```powershell
.\promtool.exe check config prometheus.yml
```

---

## 7. Stop Prometheus

Press:
```
CTRL + C
```
in the PowerShell window.

---

## 8. (Optional) Run Prometheus as a Windows Service

Prometheus does not provide a built-in Windows service.

To run as a service:
- Use tools like **NSSM (Non-Sucking Service Manager)**

---

## Notes

- Prometheus runs fully locally
- Default port is **9090**
- Configuration changes require Prometheus restart
- Data is stored in the **data** directory inside Prometheus folder
